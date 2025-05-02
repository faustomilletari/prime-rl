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
