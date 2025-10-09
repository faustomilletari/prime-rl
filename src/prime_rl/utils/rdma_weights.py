import asyncio
import os
import threading
import time
from typing import Any

import torch
from loguru import logger

# Initialize UCP before importing functions
try:
    import ucp
    # Set UCX environment variables for InfiniBand RDMA with GPU-Direct
    if 'UCX_TLS' not in os.environ:
        # Use InfiniBand transports with GPU-Direct RDMA
        # rc = InfiniBand RC (works with all RDMA devices)
        # tcp = For connection management when no IPoIB
        # cuda_copy = GPU memory operations
        # cuda_ipc = GPU IPC for same-node
        os.environ['UCX_TLS'] = 'rc,tcp,cuda_copy,cuda_ipc'
    
    # Specify InfiniBand devices (ibp0-ibp7) and eth0 for connection management
    if 'UCX_NET_DEVICES' not in os.environ:
        # Include eth0 for connection management, InfiniBand devices for data transfer
        os.environ['UCX_NET_DEVICES'] = 'ibp0:1,ibp1:1,ibp2:1,ibp3:1,ibp4:1,ibp5:1,ibp6:1,ibp7:1,eth0'
    
    # GPU-Direct RDMA settings
    if 'UCX_MEMTYPE_CACHE' not in os.environ:
        os.environ['UCX_MEMTYPE_CACHE'] = 'n'
    
    # Enable CUDA GPU-Direct RDMA
    if 'UCX_IB_GPU_DIRECT_RDMA' not in os.environ:
        os.environ['UCX_IB_GPU_DIRECT_RDMA'] = 'yes'
    
    # Allow UCX to use all devices for connection management
    if 'UCX_CM_USE_ALL_DEVICES' not in os.environ:
        os.environ['UCX_CM_USE_ALL_DEVICES'] = 'y'
    
    # Import after setting environment variables
    create_endpoint = ucp.create_endpoint
    create_listener = ucp.create_listener
    logger.info(f"UCP initialized with transports: {os.environ.get('UCX_TLS', 'default')}")
except ImportError:
    logger.warning("UCP not available, weight transfer via RDMA will not work")
    create_endpoint = None
    create_listener = None


def extract_full_weights_collective(model: torch.nn.Module) -> dict[str, torch.Tensor] | None:
    """Extract full model weights using collective operations.
    
    This function handles DTensor models where weights are sharded across ranks.
    It uses collective operations (full_tensor()) to gather complete weights.
    
    **IMPORTANT:** All ranks must call this function simultaneously for collective
    operations to work. The typical usage is:
    
        # In training loop, all ranks execute:
        full_weights = extract_full_weights_collective(model)
        
        # Only rank 0 gets the full weights:
        if rank == 0:
            weight_server.update_cached_weights(full_weights)
    
    Args:
        model: The model to extract weights from
        
    Returns:
        Full state dict on rank 0, None on other ranks
    """
    import torch.distributed as dist
    
    is_rank_0 = not dist.is_initialized() or dist.get_rank() == 0
    
    with torch.no_grad():
        state_dict = {}
        
        for name, param in model.named_parameters():
            tensor = param.data
            
            # Check if this is a DTensor that needs gathering
            if hasattr(tensor, 'full_tensor'):
                # This is a DTensor - gather full tensor (collective operation)
                full_tensor = tensor.full_tensor()
                # Only keep on rank 0
                if is_rank_0:
                    state_dict[name] = full_tensor.detach().clone().contiguous()
            elif hasattr(tensor, '_local_tensor'):
                # Fallback for older DTensor API
                full_tensor = tensor.full_tensor() if hasattr(tensor, 'full_tensor') else tensor._local_tensor
                if is_rank_0:
                    state_dict[name] = full_tensor.detach().clone().contiguous()
            else:
                # Regular tensor (not distributed)
                if is_rank_0:
                    state_dict[name] = tensor.detach().clone().contiguous()
    
    # Return full dict on rank 0, None on others
    return state_dict if is_rank_0 else None


class TrainerWeightServer:
    """UCP server on trainer (rank 0) that exposes current GPU weights for direct access."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.model = None
        self.cached_weights = None  # Cache for pre-extracted full weights
        self.lock = threading.Lock()
        self.running = False
        self.server_thread = None

    def set_model(self, model: torch.nn.Module):
        """Set the model whose weights will be served."""
        self.model = model
    
    def update_cached_weights(self, state_dict: dict[str, torch.Tensor]):
        """Update the cached weights that will be sent to inference.
        
        This should be called with full (non-sharded) weights extracted using
        collective operations where all ranks participate.
        """
        with self.lock:
            self.cached_weights = state_dict
            logger.debug(f"Updated cached weights: {len(state_dict)} parameters")

    def start(self):
        """Start the weight server in a background thread."""
        logger.info(f"Starting trainer weight server on {self.host}:{self.port}")
        
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server, name="trainer-weight-server")
        self.server_thread.daemon = True
        self.server_thread.start()

    def _run_server(self):
        """Run the async server in a thread."""
        asyncio.run(self._serve())

    async def _serve(self):
        """Main server loop handling weight transfer requests."""
        try:
            if create_listener is None:
                logger.error("UCP not available, cannot start weight server")
                return
                
            listener = create_listener(self._handle_client, port=self.port)
            logger.info(f"Trainer weight server listening on port {self.port}")
            
            while self.running:
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error in trainer weight server: {e}", exc_info=True)

    async def _handle_client(self, ep):
        """Handle a client connection for weight transfer."""
        try:
            with self.lock:
                # Use cached weights if available, otherwise extract from model
                if self.cached_weights is not None:
                    state_dict = self.cached_weights
                    logger.debug("Using cached weights for transfer")
                elif self.model is not None:
                    # Fallback: extract local shards (may not work correctly for sharded models)
                    logger.warning("No cached weights available, extracting from model (may be sharded)")
                    state_dict = {}
                    for name, param in self.model.named_parameters():
                        tensor = param.data
                        if hasattr(tensor, '_local_tensor'):
                            tensor = tensor._local_tensor
                        elif hasattr(tensor, 'to_local'):
                            tensor = tensor.to_local()
                        state_dict[name] = tensor.detach().clone().contiguous()
                else:
                    logger.error("No model or cached weights available")
                    await ep.close()
                    return
            
            # Send metadata (number of parameters)
            num_params = len(state_dict)
            total_params = sum(t.numel() for t in state_dict.values())
            logger.info(f"Sending {num_params} parameters ({total_params:,} total elements) via UCP")
            
            # Log first few parameter shapes for debugging
            for idx, (name, tensor) in enumerate(list(state_dict.items())[:5]):
                logger.debug(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
            
            # Send number of parameters
            num_params_buffer = torch.tensor([num_params], dtype=torch.int64, device='cuda')
            await ep.send(num_params_buffer)
            
            # Send each parameter name and tensor
            for idx, (name, tensor) in enumerate(state_dict.items()):
                logger.debug(f"Sending parameter {idx+1}/{num_params}: {name}, shape={tensor.shape}, dtype={tensor.dtype}, size={tensor.numel() * tensor.element_size()} bytes")
                
                # Send name length and name
                name_bytes = name.encode('utf-8')
                name_len_buffer = torch.tensor([len(name_bytes)], dtype=torch.int64, device='cuda')
                await ep.send(name_len_buffer)
                
                name_buffer = torch.frombuffer(name_bytes, dtype=torch.uint8).cuda()
                await ep.send(name_buffer)
                
                # Send tensor shape
                shape_buffer = torch.tensor(tensor.shape, dtype=torch.int64, device='cuda')
                shape_len_buffer = torch.tensor([len(shape_buffer)], dtype=torch.int64, device='cuda')
                await ep.send(shape_len_buffer)
                await ep.send(shape_buffer)
                
                # Send dtype info (as string for simplicity)
                dtype_str = str(tensor.dtype).split('.')[-1]  # e.g., "float32"
                dtype_bytes = dtype_str.encode('utf-8')
                dtype_len_buffer = torch.tensor([len(dtype_bytes)], dtype=torch.int64, device='cuda')
                await ep.send(dtype_len_buffer)
                
                dtype_buffer = torch.frombuffer(dtype_bytes, dtype=torch.uint8).cuda()
                await ep.send(dtype_buffer)
                
                # Ensure tensor is contiguous and on CUDA
                tensor = tensor.contiguous().cuda()
                
                # Send tensor with timeout handling for large tensors
                try:
                    await asyncio.wait_for(ep.send(tensor), timeout=300.0)  # 5 minute timeout
                except asyncio.TimeoutError:
                    logger.error(f"Timeout sending parameter {name}, size={tensor.numel() * tensor.element_size()} bytes")
                    raise
            
            logger.info(f"Successfully sent {num_params} weights via UCP to inference")
            await ep.close()
            
        except Exception as e:
            logger.error(f"Error handling weight transfer: {e}", exc_info=True)
            try:
                if 'ep' in locals():
                    await ep.close()
            except Exception as close_error:
                logger.error(f"Error closing endpoint: {close_error}")

    def stop(self):
        """Stop the weight server."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=5)
        logger.info("Trainer weight server stopped")


class InferenceWeightClient:
    """UCP client on inference that fetches weights directly from trainer's GPU."""

    def __init__(self, trainer_host: str, trainer_port: int, timeout: int):
        self.trainer_host = trainer_host
        self.trainer_port = trainer_port
        self.timeout = timeout

    async def fetch_weights(self) -> dict[str, torch.Tensor]:
        """Fetch weights directly from trainer's GPU via UCP."""
        try:
            if create_endpoint is None:
                raise RuntimeError("UCP not available, cannot fetch weights")
                
            logger.info(f"Connecting to trainer at {self.trainer_host}:{self.trainer_port}")
            ep = await asyncio.wait_for(
                create_endpoint(self.trainer_host, self.trainer_port),
                timeout=float(self.timeout)
            )
            
            # Receive number of parameters
            num_params_buffer = torch.empty(1, dtype=torch.int64, device='cuda')
            await ep.recv(num_params_buffer)
            num_params = num_params_buffer[0].item()
            logger.info(f"Receiving {num_params} parameters via UCP")
            
            state_dict = {}
            
            # Reusable buffers for metadata to reduce allocations
            name_len_buffer = torch.empty(1, dtype=torch.int64, device='cuda')
            shape_len_buffer = torch.empty(1, dtype=torch.int64, device='cuda')
            dtype_len_buffer = torch.empty(1, dtype=torch.int64, device='cuda')
            
            # Receive each parameter
            for idx in range(num_params):
                # Receive name
                await ep.recv(name_len_buffer)
                name_len = name_len_buffer[0].item()
                
                name_buffer = torch.empty(name_len, dtype=torch.uint8, device='cuda')
                await ep.recv(name_buffer)
                name = name_buffer.cpu().numpy().tobytes().decode('utf-8')
                del name_buffer  # Free immediately after use
                
                # Receive tensor shape
                await ep.recv(shape_len_buffer)
                shape_len = shape_len_buffer[0].item()
                
                shape_buffer = torch.empty(shape_len, dtype=torch.int64, device='cuda')
                await ep.recv(shape_buffer)
                shape = tuple(shape_buffer.cpu().numpy())
                del shape_buffer  # Free immediately after use
                
                # Receive dtype
                await ep.recv(dtype_len_buffer)
                dtype_len = dtype_len_buffer[0].item()
                
                dtype_buffer = torch.empty(dtype_len, dtype=torch.uint8, device='cuda')
                await ep.recv(dtype_buffer)
                dtype_str = dtype_buffer.cpu().numpy().tobytes().decode('utf-8')
                del dtype_buffer  # Free immediately after use
                
                # Convert dtype string to torch dtype
                dtype_map = {
                    'float32': torch.float32,
                    'float16': torch.float16,
                    'bfloat16': torch.bfloat16,
                    'int32': torch.int32,
                    'int64': torch.int64,
                }
                dtype = dtype_map.get(dtype_str, torch.float32)
                
                logger.debug(f"Receiving parameter {idx+1}/{num_params}: {name}, shape={shape}, dtype={dtype}")
                
                # Receive tensor data
                tensor = torch.empty(shape, dtype=dtype, device='cuda')
                
                # Receive with timeout handling for large tensors
                try:
                    await asyncio.wait_for(ep.recv(tensor), timeout=float(self.timeout))
                except asyncio.TimeoutError:
                    logger.error(f"Timeout receiving parameter {name}, expected size={tensor.numel() * tensor.element_size()} bytes")
                    raise
                
                state_dict[name] = tensor
            
            # Clean up metadata buffers
            del num_params_buffer, name_len_buffer, shape_len_buffer, dtype_len_buffer
            
            await ep.close()
            logger.info(f"Successfully received {len(state_dict)} weights via UCP from trainer")
            return state_dict
            
        except Exception as e:
            raise ConnectionError(f"Failed to fetch weights via UCP: {e}")