import pytest

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

# TODO(Mika): Figure out how to persist a vLLM server across integration tests


# @pytest.fixture(scope="module")
# def process(run_process: Callable[[Command, Environment], ProcessResult]) -> ProcessResult:
#     return run_process(CMD, {})
#
#
# def test_no_error(process: ProcessResult):
#     assert process.returncode == 0, f"Process failed with return code {process.returncode}"
