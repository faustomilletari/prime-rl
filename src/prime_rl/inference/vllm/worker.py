import os
import torch
import asyncio
import gc

from prime_rl.utils.rdma_weights import InferenceWeightClient


class CheckpointWorker:
    """
    This is an extension of a vLLM worker that allows for loading checkpoints
    directly from trainer's GPU via UCP. This is useful in RL training, where we want to load the
    recent policy model directly from the trainer's GPU memory.
    """

    def _get_ucp_client(self) -> InferenceWeightClient:
        """Lazy initialization of UCP client."""
        if not hasattr(self, '_ucp_client'):
            self._ucp_client = InferenceWeightClient(
                trainer_host=os.getenv('MASTER_ADDR', 'localhost'),
                trainer_port=int(os.getenv('UCP_PORT', '13337')),
                timeout=int(os.getenv('UCP_TIMEOUT', '300')),
            )
        return self._ucp_client

    def update_weights_new(self) -> None:
        """Update weights directly from trainer's GPU via UCP."""
        
        # Free GPU memory before transfer
        torch.cuda.empty_cache()
        gc.collect()
        
        ucp_client = self._get_ucp_client()

        # Stream weights directly - fetch and load one parameter at a time to minimize memory usage
        async def fetch_and_stream_weights():
            """Fetch weights and yield them one at a time to minimize memory usage."""
            async for name, tensor in ucp_client.stream_weights():
                yield name, tensor
                # Allow each tensor to be used immediately, then freed
                
        # Load weights using streaming approach
        def weights_iterator():
            # Run the async generator in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async_gen = fetch_and_stream_weights()
                while True:
                    try:
                        name, tensor = loop.run_until_complete(async_gen.__anext__())
                        if name:
                            yield name, tensor
                    except StopAsyncIteration:
                        break
            finally:
                loop.close()

        self.model_runner.model.load_weights(weights_iterator())

        # Force cleanup after loading
        gc.collect()
        torch.cuda.empty_cache()
        
        # Process weights after loading (important for some models)
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        device = next(self.model_runner.model.parameters()).device
        process_weights_after_loading(self.model_runner.model, self.model_runner.model_config, device)

    def reload_weights_new(self) -> None:
        """Reload weights (reset to base model)."""
        # For now, this is a no-op as we don't have a base model reload mechanism
        # In the future, this could reload from the original model checkpoint
        pass