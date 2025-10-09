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
    # Set UCX environment variables if not already set
    if 'UCX_TLS' not in os.environ:
        os.environ['UCX_TLS'] = 'tcp,cuda_copy,cuda_ipc'
    if 'UCX_TCP_CM_REUSEADDR' not in os.environ:
        os.environ['UCX_TCP_CM_REUSEADDR'] = 'y'
    from ucp import create_endpoint, create_listener
except ImportError:
    logger.warning("UCP not available, weight transfer via RDMA will not work")
    create_endpoint = None
    create_listener = None


class TrainerWeightServer:
    """UCP server on trainer (rank 0) that exposes current GPU weights for direct access."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.model = None
        self.lock = threading.Lock()
        self.running = False
        self.server_thread = None

    def set_model(self, model: torch.nn.Module):
        """Set the model whose weights will be served."""
        self.model = model

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
                if self.model is None:
                    await ep.close()
                    return
                
                # Get current weights directly from GPU
                state_dict = {}
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        # Handle DTensor (distributed tensor) by converting to local tensor
                        param_data = param.data
                        if hasattr(param_data, '_local_tensor'):
                            # This is a DTensor, get the local shard
                            param_data = param_data._local_tensor
                        elif hasattr(param_data, 'to_local'):
                            # Alternative DTensor API
                            param_data = param_data.to_local()
                        state_dict[name] = param_data.contiguous()
            
            # Send metadata (number of parameters)
            num_params = len(state_dict)
            logger.debug(f"Sending {num_params} parameters via UCP")
            await ep.send(torch.tensor([num_params], dtype=torch.int64, device='cuda'))
            
            # Send each parameter name and tensor
            for idx, (name, tensor) in enumerate(state_dict.items()):
                logger.debug(f"Sending parameter {idx+1}/{num_params}: {name}, shape={tensor.shape}, dtype={tensor.dtype}, size={tensor.numel() * tensor.element_size()} bytes")
                
                # Send name length and name
                name_bytes = name.encode('utf-8')
                name_len = torch.tensor([len(name_bytes)], dtype=torch.int64, device='cuda')
                await ep.send(name_len)
                await ep.send(torch.frombuffer(name_bytes, dtype=torch.uint8).to('cuda'))
                
                # Send tensor shape, dtype, and data
                shape = torch.tensor(tensor.shape, dtype=torch.int64, device='cuda')
                await ep.send(torch.tensor([len(shape)], dtype=torch.int64, device='cuda'))
                await ep.send(shape)
                
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
            num_params_tensor = torch.empty(1, dtype=torch.int64, device='cuda')
            await ep.recv(num_params_tensor)
            num_params = num_params_tensor[0].item()
            logger.info(f"Receiving {num_params} parameters via UCP")
            
            state_dict = {}
            
            # Receive each parameter
            for idx in range(num_params):
                # Receive name
                name_len_tensor = torch.empty(1, dtype=torch.int64, device='cuda')
                await ep.recv(name_len_tensor)
                name_len = name_len_tensor[0].item()
                
                name_bytes_tensor = torch.empty(name_len, dtype=torch.uint8, device='cuda')
                await ep.recv(name_bytes_tensor)
                name = name_bytes_tensor.cpu().numpy().tobytes().decode('utf-8')
                
                # Receive tensor shape and data
                shape_len_tensor = torch.empty(1, dtype=torch.int64, device='cuda')
                await ep.recv(shape_len_tensor)
                shape_len = shape_len_tensor[0].item()
                
                shape_tensor = torch.empty(shape_len, dtype=torch.int64, device='cuda')
                await ep.recv(shape_tensor)
                shape = tuple(shape_tensor.cpu().numpy())
                
                logger.debug(f"Receiving parameter {idx+1}/{num_params}: {name}, shape={shape}")
                
                # Receive tensor data (assume float32 for now, could be enhanced to receive dtype info)
                tensor = torch.empty(shape, dtype=torch.float32, device='cuda')
                
                # Receive with timeout handling for large tensors
                try:
                    await asyncio.wait_for(ep.recv(tensor), timeout=float(self.timeout))
                except asyncio.TimeoutError:
                    logger.error(f"Timeout receiving parameter {name}, expected size={tensor.numel() * tensor.element_size()} bytes")
                    raise
                
                state_dict[name] = tensor
            
            await ep.close()
            logger.info(f"Successfully received {len(state_dict)} weights via UCP from trainer")
            return state_dict
            
        except Exception as e:
            raise ConnectionError(f"Failed to fetch weights via UCP: {e}")
