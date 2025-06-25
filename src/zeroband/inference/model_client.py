from pathlib import Path

import requests


class ModelClient:
    def __init__(self, model_server_url: str):
        self.model_server_url = model_server_url

    def reload_weights(self, path: Path):
        response = requests.post(f"{self.model_server_url}/update_weights", json={"path": str(path)})
        if response.status_code != 200:
            raise Exception(f"Failed to reload model weights: {response.status_code} {response.text}")

    def get_max_batch_size(self) -> int:
        response = requests.get(f"{self.model_server_url}/max_batch_size")
        if response.status_code != 200:
            raise Exception(f"Failed to get max batch size: {response.status_code} {response.text}")
        return response.json()["max_batch_size"]
