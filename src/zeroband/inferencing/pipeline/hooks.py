import pickle
from typing import Tuple
from functools import partial

import torch
import torch.nn as nn
from prime_iroh import Node
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm import LLM

from zeroband.logger import get_logger


# TODO: Outputs of decoder blocks look different for vLLM implementations and HF-based implementations. The implementation currently breaks for HF-based implementations.
def send_intermediate_states(_, __, output: Tuple, node: Node) -> None:
    """
    A post-hook that sends the hidden states and residual of the last decoder layer to the next stage node's first layer.

    Args:
        _: The module that is being hooked
        __: The arguments to the module
        output: The output of the module (here the decoder layer output)
        node: The node that is being hooked
    """
    logger = get_logger(__name__)
    hidden_states, residual = output
    serialized_hidden_states = pickle.dumps(hidden_states.to("cpu"))
    serialized_residual = pickle.dumps(residual.to("cpu"))
    node.isend(serialized_hidden_states, tag=0, latency=None).wait()
    node.isend(serialized_residual, tag=0, latency=None).wait()
    logger.debug(
        f"Sent hidden_states and residual ({hidden_states.shape}, {residual.shape}) ({len(serialized_hidden_states) + len(serialized_residual)} bytes)"
    )


def recv_intermediate_states(_, input: Tuple, node: Node) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A pre-hook that receives the hidden states and residual from the previous stage node's last layer at the first layer of the current node.

    Assumes the node is correctly set up to receive hidden states and residual from the previous node.

    Args:
        _: The module that is being hooked
        input: The input to the module (here the positions, hidden states and residual of the previous node's last layer)
        node: The node class instances for communication
    """
    logger = get_logger(__name__)
    positions, _, kv_cache, attn_metadata, _ = input
    device = positions.device
    serialized_hidden_states = node.irecv(tag=0).wait()
    serialized_residual = node.irecv(tag=0).wait()
    hidden_states = pickle.loads(serialized_hidden_states).to(device)
    residuals = pickle.loads(serialized_residual).to(device)
    logger.debug(
        f"Got hidden_states and residuals ({hidden_states.shape}, {residuals.shape}) ({len(serialized_hidden_states) + len(serialized_residual)} bytes)"
    )

    return positions, hidden_states, kv_cache, attn_metadata, residuals


def recv_output(_, __, output, node: Node, relay=False) -> SamplerOutput:
    """
    A post-hook that receives sampling outputs from the last stage node and optionally relays them to the next stage node.
    For a pipeline with 4 stages, this hook should be registered as follows:

    Rank 1: Receive output + relay
    Rank 2: Receive output + relay
    Rank 3: Receive output
    Rank 4: *Do not register hook* (use the `send_output` hook)

    Receiving and relaying the outputs is necessary for the schedulers to be synchronized across stages.

    Args:
        _: The module that is being hooked
        __: The arguments to the module
        ____: The outputs of the module
        node: The node class instances for communication
        relay: Whether to relay the outputs to the next stage node
    """
    logger = get_logger(__name__)
    serialized_output = node.irecv(tag=0).wait()
    logger.debug(f"Received outputs ({len(serialized_output)} bytes)")
    if relay:
        node.isend(serialized_output, tag=0, latency=None).wait()
        logger.debug(f"Sent outputs ({len(serialized_output)} bytes)")
    output = pickle.loads(serialized_output)
    return output


def send_output(_, __, output: SamplerOutput, node: Node) -> None:
    """
    A post-hook that sends the sampling outputs from the last stage node to the first stage node.

    Args:
        _: The module that is being hooked
        __: The arguments to the module
        output: The outputs of the module
        node: The node class instances for communication
    """
    logger = get_logger(__name__)
    serialized_output = pickle.dumps(output)
    node.isend(serialized_output, tag=0, latency=None).wait()
    logger.debug(f"Sent outputs ({len(serialized_output)} bytes)")


def setup_hooks(stage_idx: int, num_stages: int, llm: LLM, node: Node) -> None:
    """
    Setup hooks to enable pipeline parallel inference based on pipeline topology.

    Args:
        rank: The stage index of the current process
        world_size: The total number of stages
        llm: The LLM model shard instance
        node: The node class instances for communication
    """
    assert num_stages > 1, "Pipeline parallel inference requires at least 2 stages"

    # Get logger
    logger = get_logger(__name__)

    # TODO: In vLLM v0.8.5, the sampler is moved to model runner (https://github.com/vllm-project/vllm/pull/17084)
    # Once we bump vLLM version, update this

    # Model runner owns model and sampler
    model: nn.Module = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # Extract first and last layers (pre/post-hook to recv/send intermediate states)
    first_layer: nn.Module = model.model.layers[0]
    last_layer: nn.Module = model.model.layers[-1]

    # Extract sampler (post-hook to recv/send outputs)
    sampler: nn.Module = model.sampler

    # Don't relay outputs from stage with index -2->-1
    relay = stage_idx != num_stages - 2

    if stage_idx == 0:  # First stage
        # Send intermediate states to next stage (post-hook)
        last_layer.register_forward_hook(partial(send_intermediate_states, node=node))
        logger.info("Registered post-hook send_intermediate_states on last layer")

        # Receive outputs from last stage (post-hook)
        sampler.register_forward_hook(partial(recv_output, node=node, relay=relay))
        logger.info("Registered post-hook recv_output on sampler")
    elif stage_idx == num_stages - 1:  # Last stage
        # Receive intermediate states from previous stage (pre-hook)
        first_layer.register_forward_pre_hook(partial(recv_intermediate_states, node=node))
        logger.info("Registered pre-hook recv_intermediate_states on first layer")

        # Send outputs to first  stage (post-hook)
        sampler.register_forward_hook(partial(send_output, node=node))
        logger.info("Registered post-hook send_output on sampler")
    else:
        # Receive intermediate states from previous stage and send positions to next stage (pre-hook)
        first_layer.register_forward_pre_hook(partial(recv_intermediate_states, node=node))
        logger.info("Registered pre-hook recv_intermediate_states on first layer")

        # Send intermediate states to next stage (post-hook)
        last_layer.register_forward_hook(partial(send_intermediate_states, node=node))
        logger.info("Registered post-hook send_intermediate_states on last layer")

        # Receive and relay outputs from last stage (post-hook)
        sampler.register_forward_hook(partial(recv_output, relay=relay))
        logger.info("Registered post-hook recv_output on sampler")
