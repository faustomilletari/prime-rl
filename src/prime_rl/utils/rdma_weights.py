import asyncio
import threading
import time
from typing import Any

import torch
from loguru import logger
from ucp.core import create_endpoint, create_listener


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
            listener = create_listener(self._handle_client, port=self.port)
            logger.info(f"Trainer weight server listening on port {self.port}")
            
            while self.running:
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error in trainer weight server: {e}")

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
                        state_dict[name] = param.data.contiguous()
            
            # Send metadata (number of parameters)
            num_params = len(state_dict)
            await ep.send(torch.tensor([num_params], dtype=torch.int64, device='cuda'))
            
            # Send each parameter name and tensor
            for name, tensor in state_dict.items():
                # Send name length and name
                name_bytes = name.encode('utf-8')
                name_len = torch.tensor([len(name_bytes)], dtype=torch.int64, device='cuda')
                await ep.send(name_len)
                await ep.send(torch.frombuffer(name_bytes, dtype=torch.uint8).to('cuda'))
                
                # Send tensor shape and data
                shape = torch.tensor(tensor.shape, dtype=torch.int64, device='cuda')
                await ep.send(torch.tensor([len(shape)], dtype=torch.int64, device='cuda'))
                await ep.send(shape)
                await ep.send(tensor)
            
            logger.debug("Sent weights via UCP to inference")
            await ep.close()
            
        except Exception as e:
            logger.error(f"Error handling weight transfer: {e}")
            try:
                await ep.close()
            except:
                pass

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
            ep = await create_endpoint(self.trainer_host, self.trainer_port)
            
            # Receive number of parameters
            num_params_tensor = torch.empty(1, dtype=torch.int64, device='cuda')
            await ep.recv(num_params_tensor)
            num_params = num_params_tensor[0].item()
            
            state_dict = {}
            
            # Receive each parameter
            for _ in range(num_params):
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
                
                # Receive tensor data
                tensor = torch.empty(shape, dtype=torch.float32, device='cuda')  # Will be cast appropriately
                await ep.recv(tensor)
                
                state_dict[name] = tensor
            
            await ep.close()
            logger.debug(f"Received {len(state_dict)} weights via UCP from trainer")
            return state_dict
            
        except Exception as e:
            raise ConnectionError(f"Failed to fetch weights via UCP: {e}")
