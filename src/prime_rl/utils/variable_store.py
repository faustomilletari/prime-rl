import pickle
import threading
import time
from typing import Any

import zmq
from loguru import logger


class VariableStoreServer:
    """ZeroMQ-based variable store server for distributing data to training workers."""

    def __init__(self, host: str, port: int, steps_to_preserve: int):
        self.host = host
        self.port = port
        self.steps_to_preserve = steps_to_preserve
        self.store: dict[str, bytes] = {}
        self.lock = threading.Lock()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.running = False
        self.server_thread = None

    def start(self):
        """Start the variable store server in a background thread."""
        address = f"tcp://{self.host}:{self.port}"
        self.socket.bind(address)
        logger.info(f"Variable store server started on {address}")
        
        self.running = True
        self.server_thread = threading.Thread(target=self._serve, name="variable-store-server")
        self.server_thread.daemon = True
        self.server_thread.start()

    def _serve(self):
        """Main server loop handling client requests."""
        while self.running:
            try:
                # Use polling with timeout to allow clean shutdown
                if self.socket.poll(timeout=100):
                    message = self.socket.recv()
                    request = pickle.loads(message)
                    
                    operation = request.get("operation")
                    key = request.get("key")
                    
                    if operation == "get":
                        response = self._handle_get(key)
                    elif operation == "put":
                        value = request.get("value")
                        response = self._handle_put(key, value)
                    elif operation == "delete":
                        response = self._handle_delete(key)
                    elif operation == "list_keys":
                        response = self._handle_list_keys()
                    else:
                        response = {"status": "error", "message": f"Unknown operation: {operation}"}
                    
                    self.socket.send(pickle.dumps(response))
            except Exception as e:
                logger.error(f"Error in variable store server: {e}")
                error_response = {"status": "error", "message": str(e)}
                try:
                    self.socket.send(pickle.dumps(error_response))
                except:
                    pass

    def _handle_get(self, key: str) -> dict:
        """Handle GET operation."""
        with self.lock:
            if key in self.store:
                return {"status": "success", "value": self.store[key]}
            else:
                return {"status": "error", "message": f"Key not found: {key}"}

    def _handle_put(self, key: str, value: bytes) -> dict:
        """Handle PUT operation."""
        with self.lock:
            self.store[key] = value
            return {"status": "success"}

    def _handle_delete(self, key: str) -> dict:
        """Handle DELETE operation."""
        with self.lock:
            if key in self.store:
                del self.store[key]
                return {"status": "success"}
            else:
                return {"status": "error", "message": f"Key not found: {key}"}

    def _handle_list_keys(self) -> dict:
        """Handle LIST_KEYS operation."""
        with self.lock:
            return {"status": "success", "keys": list(self.store.keys())}

    def put(self, key: str, value: Any):
        """Store a value in the variable store."""
        serialized_value = pickle.dumps(value)
        with self.lock:
            self.store[key] = serialized_value

    def delete(self, key: str):
        """Delete a value from the variable store."""
        with self.lock:
            if key in self.store:
                del self.store[key]

    def maybe_clean(self, current_step: int):
        """Clean up old variables based on current step."""
        with self.lock:
            keys_to_delete = []
            for key in self.store.keys():
                if key.startswith("step_"):
                    try:
                        step = int(key.split("_")[1].split("_")[0])
                        if current_step - step > self.steps_to_preserve:
                            keys_to_delete.append(key)
                    except (ValueError, IndexError):
                        continue
            
            for key in keys_to_delete:
                logger.debug(f"Removing old variable from store: {key}")
                del self.store[key]

    def stop(self):
        """Stop the variable store server."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=5)
        self.socket.close()
        self.context.term()
        logger.info("Variable store server stopped")


class VariableStoreClient:
    """ZeroMQ client for accessing the variable store."""

    def __init__(self, host: str, port: int, timeout: int):
        self.host = host
        self.port = port
        self.timeout = timeout * 1000  # Convert to milliseconds
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        address = f"tcp://{self.host}:{self.port}"
        
        try:
            self.socket.connect(address)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
            logger.debug(f"Variable store client connected to {address}")
        except Exception as e:
            self.socket.close()
            self.context.term()
            raise ConnectionError(f"Failed to connect to variable store at {address}: {e}")

    def get(self, key: str) -> Any:
        """Retrieve a value from the variable store."""
        request = {"operation": "get", "key": key}
        try:
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())
        except Exception as e:
            raise ConnectionError(f"Failed to communicate with variable store: {e}")
        
        if response["status"] == "success":
            return pickle.loads(response["value"])
        else:
            raise KeyError(f"Failed to get key '{key}': {response.get('message', 'Unknown error')}")

    def put(self, key: str, value: Any):
        """Store a value in the variable store."""
        serialized_value = pickle.dumps(value)
        request = {"operation": "put", "key": key, "value": serialized_value}
        try:
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())
        except Exception as e:
            raise ConnectionError(f"Failed to communicate with variable store: {e}")
        
        if response["status"] != "success":
            raise RuntimeError(f"Failed to put key '{key}': {response.get('message', 'Unknown error')}")

    def delete(self, key: str):
        """Delete a value from the variable store."""
        request = {"operation": "delete", "key": key}
        try:
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())
        except Exception as e:
            raise ConnectionError(f"Failed to communicate with variable store: {e}")
        
        if response["status"] != "success":
            raise KeyError(f"Failed to delete key '{key}': {response.get('message', 'Unknown error')}")

    def list_keys(self) -> list[str]:
        """List all keys in the variable store."""
        request = {"operation": "list_keys"}
        try:
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())
        except Exception as e:
            raise ConnectionError(f"Failed to communicate with variable store: {e}")
        
        if response["status"] == "success":
            return response["keys"]
        else:
            raise RuntimeError(f"Failed to list keys: {response.get('message', 'Unknown error')}")

    def wait_for_key(self, key: str, interval: float = 1.0, log_interval: float = 10.0):
        """Wait for a key to become available in the variable store."""
        wait_time = 0
        logger.debug(f"Waiting for key `{key}`")
        while True:
            try:
                self.get(key)
                logger.debug(f"Found key `{key}`")
                return
            except KeyError:
                if wait_time % log_interval == 0 and wait_time > 0:
                    logger.debug(f"Waiting for key `{key}` for {wait_time} seconds")
                time.sleep(interval)
                wait_time += interval

    def close(self):
        """Close the variable store client connection."""
        self.socket.close()
        self.context.term()
