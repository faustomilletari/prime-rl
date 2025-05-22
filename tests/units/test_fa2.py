import random

import pytest
import torch
from einops import rearrange
from flash_attn import flash_attn_func, flash_attn_varlen_func
from torch import nn

ERROR_ATOL = {
    torch.float: 3e-4,
    torch.half: 4e-3,
    torch.bfloat16: 2e-2,
}
ERROR_RTOL = {
    torch.float: 2e-5,
    torch.half: 4e-4,
    torch.bfloat16: 5e-3,
}


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return torch.unsqueeze(x, dim=3).expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        seqlens: torch.Tensor | None = None,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            seqlens (torch.Tensor | None): Sequence lengths tensor for packing.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        output = self.self_attention(xq, xk, xv, seqlens)

        output = output.view(bs, seqlen, -1)
        return self.wo(output)

    def self_attention(self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, seqlens: torch.Tensor | None = None) -> torch.Tensor:
        if seqlens is not None:
            return self._fa_attention_with_seqlens(xq, xk, xv, seqlens)
        else:
            return self._flash_attention(xq, xk, xv)

    def _flash_attention(self, xq, xk, xv) -> torch.Tensor:
        q = rearrange(xq, "b n t h -> b t n h")
        k = rearrange(xk, "b n t h -> b t n h")
        v = rearrange(xv, "b n t h -> b t n h")
        # q/k/b is [b, nh, t, hs] but fa2 expected [b , t, nh, hs]
        return flash_attn_func(q, k, v, causal=True)

    def _fa_attention_with_seqlens(self, xq, xk, xv, seqlens) -> torch.Tensor:
        b = xq.shape[0]
        cu_seqlens = torch.concat([torch.tensor([0]).to(xq.device), seqlens.cumsum(0)], dim=0).to(torch.int32).to(xq.device)
        max_seqlen = seqlens.max()

        q = rearrange(xq, "b n t h -> (b t) n h")
        k = rearrange(xk, "b n t h -> (b t) n h")
        v = rearrange(xv, "b n t h -> (b t) n h")
        # q/k/v is [b, nh, t, hs] but fa expected [b * t, nh, hs]

        y = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )

        y = rearrange(y, "(b t) n h -> b t n h", b=b)
        return y


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def test_sequence_packing_vs_normal_random():
    """
    take two sequences and compare the outout of attention on individual sequences vs the output of attention on the packed sequence
    """

    MAX_SEQ_LEN = 256
    DIM = 1024
    N_HEADS = 16

    model = Attention(DIM, N_HEADS).to("cuda")

    freqs_cis = precompute_freqs_cis(
        DIM // N_HEADS,
        # Need to compute until at least the max token limit for generation
        # (use 2x max sequence length to be safe)
        MAX_SEQ_LEN * 2,
        theta=10000.0,
    ).to("cuda")

    for _ in range(10):
        seq_len_cutoff = random.randint(1, MAX_SEQ_LEN)

        input_1 = torch.rand(1, seq_len_cutoff, DIM).to("cuda")
        input_2 = torch.rand(1, MAX_SEQ_LEN - seq_len_cutoff, DIM).to("cuda")

        seqlens = [seq_len_cutoff, MAX_SEQ_LEN - seq_len_cutoff]
        seqlens = torch.Tensor(seqlens).int().to("cuda")

        packed_input = torch.cat([input_1, input_2], dim=1)

        # packed output
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(packed_input, freqs_cis, seqlens=seqlens)

        output_packed_1 = output[:, :seq_len_cutoff, :]
        output_packed_2 = output[:, seq_len_cutoff:, :]

        # normal output
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_1 = model(input_1, freqs_cis)
            output_2 = model(input_2, freqs_cis)

        rtol = ERROR_RTOL[torch.bfloat16]
        atol = ERROR_ATOL[torch.bfloat16]

        ### TESTING
        assert output_1.shape == output_packed_1.shape
        assert output_2.shape == output_packed_2.shape

        torch.testing.assert_close(output_1, output_packed_1, atol=atol, rtol=rtol)
        torch.testing.assert_close(output_2, output_packed_2, atol=atol, rtol=rtol)


@pytest.mark.parametrize("seq_len, head_dim, nheads", [(8, 64, 1), (16, 64, 2)])
def test_flashattn_equivalence_on_subsequence(seq_len, head_dim, nheads):
    torch.manual_seed(42)
    dtype = torch.float16
    device = "cuda"

    batch_size = 1
    d_model = nheads * head_dim

    # Input (1, S, d_model)
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    # Projections
    proj_q = torch.nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
    proj_k = torch.nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
    proj_v = torch.nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

    q = proj_q(x).reshape(batch_size * seq_len, nheads, head_dim)
    k = proj_k(x).reshape(batch_size * seq_len, nheads, head_dim)
    v = proj_v(x).reshape(batch_size * seq_len, nheads, head_dim)

    cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    # FlashAttention on (1, S)
    out_ref = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
    )

    # Double input along sequence dimension -> (1, 2S, d_model)
    x2 = torch.cat([x, x], dim=1)
    total_len = 2 * seq_len

    q2 = proj_q(x2).reshape(batch_size * total_len, nheads, head_dim)
    k2 = proj_k(x2).reshape(batch_size * total_len, nheads, head_dim)
    v2 = proj_v(x2).reshape(batch_size * total_len, nheads, head_dim)

    cu_seqlens_q2 = torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32, device=device)
    cu_seqlens_k2 = torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32, device=device)

    # FlashAttention on (1, 2S)
    out_2S = flash_attn_varlen_func(
        q=q2,
        k=k2,
        v=v2,
        cu_seqlens_q=cu_seqlens_q2,
        cu_seqlens_k=cu_seqlens_k2,
        max_seqlen_q=total_len,
        max_seqlen_k=total_len,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
    )

    # Slice only the first S positions from out_2S
    out_2S_sliced_left = out_2S[:seq_len]
    out_2S_sliced_right = out_2S[seq_len : 2 * seq_len]

    torch.testing.assert_close(out_ref, out_2S_sliced_left)
    torch.testing.assert_close(out_ref, out_2S_sliced_right)
