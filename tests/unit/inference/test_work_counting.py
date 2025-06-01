import pytest
from transformers import AutoConfig

from zeroband.inference.work_counting import get_inference_input_output_flops_deepseek_v3, get_inference_input_output_flops_qwen3


@pytest.mark.parametrize(
    "model_name_or_path, active_params",
    [
        ("deepseek-ai/DeepSeek-R1-0528", 37e9),  # 37B Active Params from https://arxiv.org/pdf/2412.19437
    ],
)
def test_get_inference_input_output_flops_deepseek_v3(model_name_or_path: str, active_params: int):
    config = AutoConfig.from_pretrained(model_name_or_path)

    # 1 input token, 0 output tokens should be almost equal to 2 * active params
    input_flops, output_flops = get_inference_input_output_flops_deepseek_v3(config, 1, 0)
    assert abs(input_flops - 2 * active_params) / active_params < 0.05


@pytest.mark.parametrize(
    "model_name_or_path, active_params",
    [
        ("Qwen/Qwen3-0.6B", 0.6e9),
        ("Qwen/Qwen3-1.7B", 1.7e9),
        ("Qwen/Qwen3-4B", 4e9),
        ("Qwen/Qwen3-8B", 7.6e9),  # This is only 8B because it has untied embs somehow
        ("Qwen/Qwen3-14B", 14e9),
        ("Qwen/Qwen3-32B", 32e9),
    ],
)
def test_get_inference_input_output_flops_qwen3(model_name_or_path: str, active_params: int):
    config = AutoConfig.from_pretrained(model_name_or_path)
    input_flops, output_flops = get_inference_input_output_flops_qwen3(config, 1, 0)
    assert abs(input_flops - 2 * active_params) / active_params < 0.05
