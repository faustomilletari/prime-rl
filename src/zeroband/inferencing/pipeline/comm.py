import time
from prime_iroh import Node
from typing import Optional

from zeroband.logger import get_logger


def setup_comm(num_stages: int, iroh_seed: Optional[int], iroh_peer_id: Optional[str]) -> Node:
    assert num_stages > 1, "Pipeline parallel inference requires at least 2 stages"

    # Get logger
    logger = get_logger(__name__)

    # Setup node (with or without seed)
    if iroh_seed is not None:
        # If seed is provided, create a new node with the seed
        node = Node.with_seed(num_streams=1, seed=iroh_seed)
    else:
        # If no seed, create a new node
        node = Node(num_streams=1)
    logger.info(f"Created node (ID={node.node_id()})")

    # Connect to peer
    if iroh_peer_id is None:
        iroh_peer_id = input("Enter Peer ID: ").strip()
    logger.info(f"Setting up outgoing connection to {iroh_peer_id}")
    node.connect(iroh_peer_id)
    logger.info(f"Outgoing connection to {iroh_peer_id} successful!")

    # Wait for connection to sender and receiver to be established
    # Note: This requires the PP communication loop to be closed, e.g. for 4 stages:
    # 0 -> 1 -> 2 -> 3 -> 0
    logger.info("Waiting for incoming connection...")
    while not node.is_ready():
        time.sleep(0.1)
    logger.info("All connections successful!")

    return node


if __name__ == "__main__":
    import os
    from multiprocessing import Process

    # Pre-computed node IDs for differen seeds (useful for debugging)
    IROH_NODE_ID_MAP = {
        0: "ee1aa49a4459dfe813a3cf6eb882041230c7b2558469de81f87c9bf23bf10a03",
        1: "ff87a0b0a3c7c0ce827e9cada5ff79e75a44a0633bfcb5b50f99307ddb26b337",
        2: "191fc38f134aaf1b7fdb1f86330b9d03e94bd4ba884f490389de964448e89b3f",
        3: "c5bbbb60e412879bbec7bb769804fa8e36e68af10d5477280b63deeaca931bed",
        4: "4f44e6c7bdfed3d9f48d86149ee3d29382cae8c83ca253e06a70be54a301828b",
        5: "e2e8aa145e1ec5cb01ebfaa40e10e12f0230c832fd8135470c001cb86d77de00",
        6: "17888c2ca502371245e5e35d5bcf35246c3bc36878e859938c9ead3c54db174f",
        7: "478243aed376da313d7cf3a60637c264cb36acc936efb341ff8d3d712092d244",
    }

    def run_stage(rank: int, world_size: int):
        # Set environment variables
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(0)
        os.environ["LOCAL_WORLD_SIZE"] = str(1)

        from zeroband.inferencing.pipeline.utils import IROH_NODE_ID_MAP

        iroh_peer_id = IROH_NODE_ID_MAP[(rank + 1) % world_size]
        setup_comm(num_stages=world_size, iroh_seed=rank, iroh_peer_id=iroh_peer_id)

    processes = []
    world_size = 2
    for rank in range(world_size):
        p = Process(target=run_stage, args=(rank, world_size))
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()
