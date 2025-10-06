import asyncio
import threading
import time
from typing import Any

import torch
import ucp
from loguru import logger


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
            listener = ucp.create_listener(self._handle_client, port=self.port)
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
            ep = await ucp.create_endpoint(self.trainer_host, self.trainer_port)
            
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


class WeightSyncCoordinator:
    """Simple coordinator that manages weight synchronization between trainer and orchestrator."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.current_step = 0
        self.weights_ready = False
        self.lock = threading.Lock()
        self.context = None
        self.socket = None
        self.running = False
        self.server_thread = None

    def start(self):
        """Start the weight sync coordinator."""
        import zmq
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        address = f"tcp://{self.host}:{self.port}"
        self.socket.bind(address)
        logger.info(f"Weight sync coordinator started on {address}")
        
        self.running = True
        self.server_thread = threading.Thread(target=self._serve, name="weight-sync-coordinator")
        self.server_thread.daemon = True
        self.server_thread.start()

    def _serve(self):
        """Main coordinator loop."""
        import pickle
        
        while self.running:
            try:
                if self.socket.poll(timeout=100):
                    message = self.socket.recv()
                    request = pickle.loads(message)
                    
                    if request.get("operation") == "weights_ready":
                        step = request.get("step")
                        with self.lock:
                            self.current_step = step
                            self.weights_ready = True
                        response = {"status": "success"}
                    elif request.get("operation") == "check_weights_ready":
                        with self.lock:
                            response = {
                                "status": "success",
                                "step": self.current_step,
                                "ready": self.weights_ready
                            }
                    else:
                        response = {"status": "error", "message": "Unknown operation"}
                    
                    self.socket.send(pickle.dumps(response))
            except Exception as e:
                logger.error(f"Error in weight sync coordinator: {e}")

    def stop(self):
        """Stop the weight sync coordinator."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=5)
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        logger.info("Weight sync coordinator stopped")


class WeightSyncClient:
    """Client for communicating with the weight sync coordinator."""

    def __init__(self, host: str, port: int, timeout: int):
        self.host = host
        self.port = port
        self.timeout = timeout
        import zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        address = f"tcp://{self.host}:{self.port}"
        
        try:
            self.socket.connect(address)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout * 1000)
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout * 1000)
            logger.debug(f"Weight sync client connected to {address}")
        except Exception as e:
            self.socket.close()
            self.context.term()
            raise ConnectionError(f"Failed to connect to weight sync coordinator at {address}: {e}")

    def signal_weights_ready(self, step: int):
        """Signal that weights are ready for a given step."""
        import pickle
        
        request = {"operation": "weights_ready", "step": step}
        try:
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())
            if response["status"] != "success":
                raise RuntimeError(f"Failed to signal weights ready: {response.get('message', 'Unknown error')}")
        except Exception as e:
            raise ConnectionError(f"Failed to signal weights ready: {e}")

    def check_weights_ready(self) -> tuple[int, bool]:
        """Check if weights are ready and return (step, ready)."""
        import pickle
        
        request = {"operation": "check_weights_ready"}
        try:
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())
            if response["status"] != "success":
                raise RuntimeError(f"Failed to check weights ready: {response.get('message', 'Unknown error')}")
            return response["step"], response["ready"]
        except Exception as e:
            raise ConnectionError(f"Failed to check weights ready: {e}")

    def wait_for_weights(self, step: int, interval: float = 1.0, log_interval: float = 10.0):
        """Wait for weights to be ready for a given step."""
        wait_time = 0
        logger.debug(f"Waiting for weights at step {step}")
        while True:
            try:
                current_step, ready = self.check_weights_ready()
                if ready and current_step >= step:
                    logger.debug(f"Found weights ready for step {step}")
                    return
            except Exception as e:
                logger.debug(f"Error checking weights ready: {e}")
            
            if wait_time % log_interval == 0 and wait_time > 0:
                logger.debug(f"Waiting for weights at step {step} for {wait_time} seconds")
            time.sleep(interval)
            wait_time += interval

    def close(self):
        """Close the weight sync client connection."""
        self.socket.close()
        self.context.term()
