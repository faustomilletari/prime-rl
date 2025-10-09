import os

from prime_rl.utils.rdma_weights import InferenceWeightClient



class CheckpointWorker:
    """
    This is an extension of a vLLM worker that allows for loading checkpoints
    directly from trainer's GPU via UCP. This is useful in RL training, where we want to load the
    recent policy model directly from the trainer's GPU memory.
    """

    def __init__(self, args, kwargs):
        self.ucp_client = InferenceWeightClient(
            trainer_host=os.getenv('MASTER_ADDR'),
        )

    def update_weights_new(self) -> None:
        """Update weights directly from trainer's GPU via UCP."""
        if self.ucp_client is None:
            raise RuntimeError("UCP client not initialized")

        # Get weights directly from trainer's GPU via UCP
        import asyncio
        gpu_state_dict = asyncio.run(self.ucp_client.fetch_weights())

        def weights_iterator():
            for key, value in gpu_state_dict.items():
                if not key:
                    continue
                yield key, value  # Already on GPU, no .cuda() needed

        self.model_runner.model.load_weights(weights_iterator())

        # Process weights after loading (important for some models)
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        device = next(self.model_runner.model.parameters()).device
        process_weights_after_loading(self.model_runner.model, self.model_runner.model_config, device)

    def reload_weights_new(self) -> None:
        """Reload weights (reset to base model)."""
        # For now, this is a no-op as we don't have a base model reload mechanism
        # In the future, this could reload from the original model checkpoint
        pass
