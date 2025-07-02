import asyncio
import os
import subprocess
import time
import urllib.error
import urllib.request

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

CMD = ["uv", "run", "infer", "@configs/inference/debug.toml"]


async def wait_for_server_health(base_url: str, timeout: int = 60, interval: int = 1) -> bool:
    """Wait for the server to be healthy by checking the /health endpoint."""
    health_url = f"{base_url}/health"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        await asyncio.sleep(interval)

    return False


def test_vllm_server_startup_and_shutdown():
    """Test that the vLLM server starts up, is reachable, and shuts down properly."""
    # Start the server as a subprocess
    env = dict(os.environ)
    process = subprocess.Popen(CMD, env=env)

    try:
        # Wait for the server to be healthy
        # Default port is 8000 based on the config
        base_url = "http://localhost:8000"
        is_healthy = asyncio.run(wait_for_server_health(base_url, timeout=60))

        assert is_healthy, "vLLM server did not become healthy within timeout"

        # Optionally, test that we can also reach the models endpoint
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as response:
                assert response.status == 200, f"Failed to reach models endpoint: {response.status}"
        except Exception as e:
            pytest.fail(f"Failed to reach models endpoint: {e}")

    finally:
        # Shut down the server gracefully
        process.terminate()

        # Wait for the process to terminate (with timeout)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # If it doesn't terminate gracefully, kill it
            process.kill()
            process.wait()

        assert process.returncode is not None, "Process did not terminate properly"
