from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

# Note: Only matmuls are counted


def get_inference_input_output_flops_qwen3(config: Qwen3Config, input_tokens: int, output_tokens: int) -> tuple[int, int]:
    """Get input and output flops for Qwen3 inference"""
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    head_dim = config.head_dim
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    num_hidden_layers = config.num_hidden_layers

    # Linears
    ## Attn
    q_flops = 2 * num_hidden_layers * hidden_size * num_attention_heads * head_dim
    k_flops = 2 * num_hidden_layers * hidden_size * num_key_value_heads * head_dim
    v_flops = 2 * num_hidden_layers * hidden_size * num_key_value_heads * head_dim
    o_flops = 2 * num_hidden_layers * hidden_size * num_attention_heads * head_dim
    ## MLP
    mlp_flops = 2 * num_hidden_layers * 3 * intermediate_size * hidden_size
    ## LM Head
    lm_head_flops = 2 * vocab_size * hidden_size
    ## Total
    input_linear_flops = (q_flops + k_flops + v_flops + o_flops + mlp_flops + lm_head_flops) * input_tokens
    output_linear_flops = (q_flops + k_flops + v_flops + o_flops + mlp_flops + lm_head_flops) * output_tokens

    # SDPA
    ## 4lhqt from mm
    input_sdpa = 4 * num_hidden_layers * head_dim * num_attention_heads * ((input_tokens + 1) * input_tokens // 2)
    output_sdpa = 4 * num_hidden_layers * head_dim * num_attention_heads * ((output_tokens + input_tokens + 1) * output_tokens // 2)

    return input_linear_flops + input_sdpa, output_linear_flops + output_sdpa


def get_inference_input_output_flops_deepseek_v3(config: DeepseekV3Config, input_tokens: int, output_tokens: int) -> tuple[int, int]:
    """Get input and output flops for Deepseek V3 inference"""
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    head_dim = config.qk_head_dim  # Nope + Rope included
    num_attention_heads = config.num_attention_heads
    num_hidden_layers = config.num_hidden_layers

    # MoE
    num_dense_layers = config.first_k_dense_replace
    num_sparse_layers = config.num_hidden_layers - num_dense_layers
    shared_experts = config.n_shared_experts
    routed_experts = config.n_routed_experts
    experts_per_tok = config.num_experts_per_tok
    intermediate_size = config.intermediate_size
    moe_intermediate_size = config.moe_intermediate_size

    # Linears
    ## Attn
    q_flops = 2 * num_hidden_layers * (hidden_size * config.q_lora_rank + config.q_lora_rank * num_attention_heads * config.qk_head_dim)
    kv_flops = (
        2
        * num_hidden_layers
        * (
            hidden_size * (config.kv_lora_rank + config.qk_rope_head_dim)
            + config.kv_lora_rank * num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim)
        )
    )
    o_flops = 2 * num_hidden_layers * (num_attention_heads * config.v_head_dim * hidden_size)
    ## MLP
    dense_mlp_flops = 2 * num_dense_layers * 3 * intermediate_size * hidden_size
    sparse_mlp_flops = num_sparse_layers * (
        2 * shared_experts * 3 * moe_intermediate_size * hidden_size  # Shared experts
        + 2 * experts_per_tok * 3 * moe_intermediate_size * hidden_size  # Routed experts
        + 2 * routed_experts * hidden_size  # Router
    )
    ## LM Head
    lm_head_flops = 2 * vocab_size * hidden_size
    ## Total
    input_linear_flops = (q_flops + kv_flops + o_flops + dense_mlp_flops + sparse_mlp_flops + lm_head_flops) * input_tokens
    output_linear_flops = (q_flops + kv_flops + o_flops + dense_mlp_flops + sparse_mlp_flops + lm_head_flops) * output_tokens

    # SDPA
    ## 4lhqt from mm
    input_sdpa = 4 * num_hidden_layers * head_dim * num_attention_heads * ((input_tokens + 1) * input_tokens // 2)
    output_sdpa = 4 * num_hidden_layers * head_dim * num_attention_heads * ((output_tokens + input_tokens + 1) * output_tokens // 2)

    return input_linear_flops + input_sdpa, output_linear_flops + output_sdpa
