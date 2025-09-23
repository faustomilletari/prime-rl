import asyncio
import pickle
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import zmq
import zmq.asyncio
from torch import Tensor

from prime_rl.utils.logger import get_logger


class MessageType(Enum):
    STORE = "store"
    RETRIEVE = "retrieve"
    DELETE = "delete"
    LIST = "list"
    EXISTS = "exists"


class DataStoreServer:
    """
    ZeroMQ server that acts as a distributed store for data.
    Supports store, retrieve, delete, list, and exists operations.
    """

    def __init__(self, port: int = 5555):
        self.port = port
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REP)
        self.data_store: Dict[str, Any] = {}
        self.running = False
        self._logger = get_logger()

    async def start(self):
        """Start the data store server."""
        self.socket.bind(f"tcp://*:{self.port}")
        self.running = True
        self._logger.info(f"Data store server started on port {self.port}")

        while self.running:
            try:
                # Receive request
                message = await self.socket.recv()
                request = pickle.loads(message)

                # Process request
                response = self._handle_request(request)

                # Send response
                response_data = pickle.dumps(response)
                await self.socket.send(response_data)

            except zmq.Again:
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                continue
            except Exception as e:
                self._logger.error(f"Error handling request: {e}")
                error_response = {"status": "error", "message": str(e)}
                await self.socket.send(pickle.dumps(error_response))

    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming requests based on message type."""
        msg_type = MessageType(request.get("type"))
        store_key = request.get("store_key")

        if msg_type == MessageType.STORE:
            store_data = request.get("store_data")
            self.data_store[store_key] = store_data
            self._logger.debug(f"Stored data '{store_key}'")
            return {"status": "success", "message": f"Data '{store_key}' stored"}

        elif msg_type == MessageType.RETRIEVE:
            if store_key in self.data_store:
                store_data = self.data_store[store_key]
                self._logger.debug(f"Retrieved data '{store_key}'")
                return {
                    "status": "success",
                    "store_data": store_data,
                }
            else:
                return {"status": "error", "message": f"Data '{store_key}' not found"}

        elif msg_type == MessageType.DELETE:
            if store_key in self.data_store:
                self.data_store[store_key] = None
                self._logger.debug(f"Deleted data '{store_key}'")
                return {"status": "success", "message": f"Data '{store_key}' removed from store"}
            else:
                return {"status": "error", "message": f"Data '{store_key}' not found"}

        elif msg_type == MessageType.LIST:
            data_list = list(self.data_store.keys())
            return {"status": "success", "data_keys": data_list}

        elif msg_type == MessageType.EXISTS:
            exists = store_key in self.data_store
            return {"status": "success", "exists": exists}

        else:
            return {"status": "error", "message": f"Unknown message type: {msg_type}"}

    async def stop(self):
        """Stop the server."""
        self.running = False
        self.socket.close()
        self.context.term()
        self._logger.info("Data store server stopped")


class DataStoreClient:
    """
    ZeroMQ client for interacting with the data store server.
    Provides methods to store, retrieve, delete, list, and check existence of data.
    """

    def __init__(self, server_address: str = "localhost", server_port: int = 5555, timeout: int = 30000):
        self.server_address = server_address
        self.server_port = server_port
        self.timeout = timeout
        self.context = zmq.asyncio.Context()
        self.socket = None
        self._logger = get_logger()

    async def _connect(self):
        """Connect to the data store server."""
        if self.socket:
            self.socket.close()

        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
        self.socket.connect(f"tcp://{self.server_address}:{self.server_port}")
        self._logger.debug(f"Connected to data store at {self.server_address}:{self.server_port}")

    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to server and return response."""
        if not self.socket:
            await self._connect()

        try:
            # Send request
            message = pickle.dumps(request)
            await self.socket.send(message)

            # Receive response
            response_data = await self.socket.recv()
            response = pickle.loads(response_data)
            return response

        except zmq.Again:
            self._logger.error("Request timeout - reconnecting...")
            await self._connect()
            raise TimeoutError("Request timed out")
        except Exception as e:
            self._logger.error(f"Communication error: {e}")
            await self._connect()
            raise

    async def store_data(self, store_key: str, store_data: Any) -> bool:
        """
        Store data on the server.

        Args:
            store_key: Unique identifier for the data (e.g., "step_123_rank_0")
            store_data: Data to store

        Returns:
            bool: True if successful, False otherwise
        """
        request = {
            "type": MessageType.STORE.value,
            "store_key": store_key,
            "store_data": store_data
        }

        try:
            response = await self._send_request(request)
            success = response.get("status") == "success"

            if success:
                self._logger.debug(f"Successfully stored data '{store_key}'")
            else:
                self._logger.error(f"Failed to store data '{store_key}': {response.get('message')}")

            return success
        except Exception as e:
            self._logger.error(f"Failed to store data '{store_key}': {e}")
            return False

    async def retrieve_data(self, store_key: str) -> Optional[Any]:
        """
        Retrieve data from the server.

        Args:
            store_key: Unique identifier for the data

        Returns:
            Any or None: Retrieved data or None if not found
        """
        request = {
            "type": MessageType.RETRIEVE.value,
            "store_key": store_key
        }

        try:
            response = await self._send_request(request)

            if response.get("status") == "success":
                store_data = response.get("store_data")
                self._logger.debug(f"Successfully retrieved data '{store_key}'")
                return store_data
            else:
                self._logger.error(f"Failed to retrieve data '{store_key}': {response.get('message')}")
                return None
        except Exception as e:
            self._logger.error(f"Failed to retrieve data '{store_key}': {e}")
            return None

    async def delete_data(self, store_key: str) -> bool:
        """
        Delete data from the server.

        Args:
            store_key: Unique identifier for the data

        Returns:
            bool: True if successful, False otherwise
        """
        request = {
            "type": MessageType.DELETE.value,
            "store_key": store_key
        }

        try:
            response = await self._send_request(request)
            success = response.get("status") == "success"

            if success:
                self._logger.debug(f"Successfully deleted data '{store_key}'")
            else:
                self._logger.error(f"Failed to delete data '{store_key}': {response.get('message')}")

            return success
        except Exception as e:
            self._logger.error(f"Failed to delete data '{store_key}': {e}")
            return False

    async def list_data(self) -> Optional[list]:
        """
        List all data stored on the server.

        Returns:
            list: List of data keys or None if error
        """
        request = {"type": MessageType.LIST.value}

        try:
            response = await self._send_request(request)

            if response.get("status") == "success":
                data_keys = response.get("data_keys", [])
                self._logger.debug(f"Found {len(data_keys)} data entries on server")
                return data_keys
            else:
                self._logger.error(f"Failed to list data: {response.get('message')}")
                return None
        except Exception as e:
            self._logger.error(f"Failed to list data: {e}")
            return None

    async def data_exists(self, store_key: str) -> bool:
        """
        Check if data exists on the server.

        Args:
            store_key: Unique identifier for the data

        Returns:
            bool: True if data exists, False otherwise
        """
        request = {
            "type": MessageType.EXISTS.value,
            "store_key": store_key
        }

        try:
            response = await self._send_request(request)

            if response.get("status") == "success":
                exists = response.get("exists", False)
                self._logger.debug(f"Data '{store_key}' {'exists' if exists else 'does not exist'}")
                return exists
            else:
                self._logger.error(f"Failed to check data existence: {response.get('message')}")
                return False
        except Exception as e:
            self._logger.error(f"Failed to check data existence: {e}")
            return False

    async def close(self):
        """Close the client connection."""
        if self.socket:
            self.socket.close()
        self.context.term()
        self._logger.debug("Data store client closed")


class SyncDataStoreClient:
    """
    Synchronous version of DataStoreClient for use in non-async contexts.
    """

    def __init__(self, server_address: str = "localhost", server_port: int = 5555, timeout: int = 30000):
        self.server_address = server_address
        self.server_port = server_port
        self.timeout = timeout
        self.context = zmq.Context()
        self.socket = None
        self._logger = get_logger()
        self._connect()

    def _connect(self):
        """Connect to the data store server."""
        if self.socket:
            self.socket.close()

        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
        self.socket.connect(f"tcp://{self.server_address}:{self.server_port}")
        self._logger.debug(f"Connected to data store at {self.server_address}:{self.server_port}")

    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to server and return response."""
        try:
            # Send request
            message = pickle.dumps(request)
            self.socket.send(message)

            # Receive response
            response_data = self.socket.recv()
            response = pickle.loads(response_data)
            return response

        except zmq.Again:
            self._logger.error("Request timeout - reconnecting...")
            self._connect()
            raise TimeoutError("Request timed out")
        except Exception as e:
            self._logger.error(f"Communication error: {e}")
            self._connect()
            raise

    def store_data(self, store_key: str, store_data: Any) -> bool:
        """
        Store data on the server.

        Args:
            store_key: Unique identifier for the data (e.g., "step_123_rank_0")
            store_data: Data to store

        Returns:
            bool: True if successful, False otherwise
        """
        request = {
            "type": MessageType.STORE.value,
            "store_key": store_key,
            "store_data": store_data
        }

        try:
            response = self._send_request(request)
            success = response.get("status") == "success"

            if success:
                self._logger.debug(f"Successfully stored data '{store_key}'")
            else:
                self._logger.error(f"Failed to store data '{store_key}': {response.get('message')}")

            return success
        except Exception as e:
            self._logger.error(f"Failed to store data '{store_key}': {e}")
            return False

    def retrieve_data(self, store_key: str) -> Optional[Any]:
        """
        Retrieve data from the server.

        Args:
            store_key: Unique identifier for the data

        Returns:
            Any or None: Retrieved data or None if not found
        """
        request = {
            "type": MessageType.RETRIEVE.value,
            "store_key": store_key
        }

        try:
            response = self._send_request(request)

            if response.get("status") == "success":
                store_data = response.get("store_data")
                self._logger.debug(f"Successfully retrieved data '{store_key}'")
                return store_data
            else:
                self._logger.debug(f"Failed to retrieve data '{store_key}': {response.get('message')}")
                return None
        except Exception as e:
            self._logger.error(f"Failed to retrieve data '{store_key}': {e}")
            return None

    def delete_data(self, store_key: str) -> bool:
        """
        Delete data from the server.

        Args:
            store_key: Unique identifier for the data

        Returns:
            bool: True if successful, False otherwise
        """
        request = {
            "type": MessageType.DELETE.value,
            "store_key": store_key
        }

        try:
            response = self._send_request(request)
            success = response.get("status") == "success"

            if success:
                self._logger.debug(f"Successfully deleted data '{store_key}'")
            else:
                self._logger.error(f"Failed to delete data '{store_key}': {response.get('message')}")

            return success
        except Exception as e:
            self._logger.error(f"Failed to delete data '{store_key}': {e}")
            return False

    def data_exists(self, store_key: str) -> bool:
        """
        Check if data exists on the server.

        Args:
            store_key: Unique identifier for the data

        Returns:
            bool: True if data exists, False otherwise
        """
        request = {
            "type": MessageType.EXISTS.value,
            "store_key": store_key
        }

        try:
            response = self._send_request(request)

            if response.get("status") == "success":
                exists = response.get("exists", False)
                self._logger.debug(f"Data '{store_key}' {'exists' if exists else 'does not exist'}")
                return exists
            else:
                self._logger.error(f"Failed to check data existence: {response.get('message')}")
                return False
        except Exception as e:
            self._logger.error(f"Failed to check data existence: {e}")
            return False

    def close(self):
        """Close the client connection."""
        if self.socket:
            self.socket.close()
        self.context.term()
        self._logger.debug("Data store client closed")


async def wait_for_data(client: DataStoreClient, store_key: str, interval: float = 1.0, log_interval: int = 10) -> None:
    """
    Wait for data to become available on the server.

    Args:
        client: DataStoreClient instance
        store_key: Key of the data to wait for
        interval: Time to wait between checks (seconds)
        log_interval: How often to log waiting status (in check cycles)
    """
    logger = get_logger()
    wait_cycles = 0
    logger.debug(f"Waiting for data '{store_key}'")

    while True:
        if await client.data_exists(store_key):
            logger.debug(f"Found data '{store_key}'")
            break

        if wait_cycles % log_interval == 0 and wait_cycles > 0:
            logger.debug(f"Waiting for data '{store_key}' for {wait_cycles * interval:.1f} seconds")

        await asyncio.sleep(interval)
        wait_cycles += 1


def wait_for_data_sync(client: SyncDataStoreClient, store_key: str, interval: float = 1.0, log_interval: int = 10) -> None:
    """
    Synchronous version of wait_for_data.

    Args:
        client: SyncDataStoreClient instance
        store_key: Key of the data to wait for
        interval: Time to wait between checks (seconds)
        log_interval: How often to log waiting status (in check cycles)
    """
    logger = get_logger()
    wait_cycles = 0
    logger.debug(f"Waiting for data '{store_key}'")

    while True:
        if client.data_exists(store_key):
            logger.debug(f"Found data '{store_key}'")
            break

        if wait_cycles % log_interval == 0 and wait_cycles > 0:
            logger.debug(f"Waiting for data '{store_key}' for {wait_cycles * interval:.1f} seconds")

        time.sleep(interval)
        wait_cycles += 1


# Backward compatibility aliases
RolloutStoreServer = DataStoreServer
RolloutStoreClient = DataStoreClient
SyncRolloutStoreClient = SyncDataStoreClient
wait_for_rollout = wait_for_data
wait_for_rollout_sync = wait_for_data_sync