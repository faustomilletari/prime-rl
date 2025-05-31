import os
import json
from functools import cache

from prime_code_sandbox import (
    make_sandbox_request,
    CodeSandbox,
    SandboxRequest,
    CodeRunArgs,
)

from zeroband.utils.logger import get_logger
from zeroband.inference.genesys.code import extract_answer

@cache
def get_sandbox() -> CodeSandbox:
    base_url: str | None = os.environ.get("PRIME_CODE_SANDBOX_URL", None)
    port = int(os.environ.get("PRIME_CODE_SANDBOX_PORT", 8000))
    auth_token = os.environ.get("PRIME_CODE_SANDBOX_AUTH_TOKEN", None)
    return CodeSandbox(base_url=base_url, port=port, auth_token=auth_token)

def evaluate_code(completion: str, verification_info: dict) -> int:
    solution = extract_answer(completion)
    if solution is None:
        return 0

    test_cases = json.loads(verification_info["test_cases"])
    inputs = test_cases["inputs"]
    outputs = test_cases["outputs"]
    fn_name: str | None = verification_info.get("fn_name", None)

    results = run_tests(completion, inputs, outputs, fn_name)
    return 1 if all(results) else 0

def run_tests(solution: str, inputs: list, outputs: list, fn_name: str | None) -> list[bool]:
    assert len(inputs) == len(outputs), "Inputs and outputs must have the same length"
    sandbox = get_sandbox()
    
    # TODO: Run the code, return the result
    return []


def convert_testcase_to_requests(
    completion: str, input_data: str, output_data: str, fn_name: str | None
) -> list[SandboxRequest]:
    # TODO: Figure out what to do here.
    if fn_name is None:
        code_to_run = completion
    else:
        code_to_run = completion

    requests = []
    for i in range(len(input_data)):
        args: CodeRunArgs = CodeRunArgs(code=code_to_run)
        requests.append(SandboxRequest(args=args, language="python"))
    return requests
