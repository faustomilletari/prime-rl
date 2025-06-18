from prime_code_sandbox import (
    CodeRunArgs,
    CodeSandbox,
    SandboxRequest,
    SandboxRequestBatch,
    SandboxResponse,
    SandboxResponseBatch,
    make_sandbox_request_batch,
)


def codeforces_reward(code: str, verification_info: dict, verbose=False) -> float:
    language = verification_info["language"]
    input_mode = verification_info["input_mode"]
    time_limit = verification_info["time_limit"]
    #memory_limit = verification_info["memory_limit"] # TODO: Implement memory limit in code sandbox (cgroups agony)
    tests = verification_info["tests"]

    local_sandbox = CodeSandbox()

    def create_request(code: str, test: dict) -> SandboxRequest:
        if input_mode == "stdio":
            return SandboxRequest(
                args=CodeRunArgs(code=code, run_timeout=time_limit, stdin=test["input"]),
                language=language,
            )
        elif input_mode == "file":
            return SandboxRequest(
                args=CodeRunArgs(code=code, run_timeout=time_limit, files={"input.txt": test["input"]}, fetch_files=["output.txt"]),
                language=language,
            )
        else:
            raise NotImplementedError(f"input mode doesn't exist: {input_mode}")


    requests = SandboxRequestBatch(
        requests=[create_request(code, test) for test in tests]
    )

    responses: SandboxResponseBatch = make_sandbox_request_batch(
        sandbox=local_sandbox, request_batch=requests
    )

    assert len(responses.responses) == len(tests), "Number of responses does not match number of tests."

    def extract_response_text(response: SandboxResponse) -> str | None:
        try:
            if input_mode == "stdio":
                response_text = response.result.run_result.stdout.strip() # type: ignore (Throw exception on missing attr)
            elif input_mode == "file":
                response_text = response.result.files["output.txt"].strip()
            else:
                return None
        except Exception:
            return None
        return response_text

    def score_response(response: SandboxResponse, test: dict) -> bool:
        return extract_response_text(response) == test["output"].strip()

    scores: list[bool] = [
        score_response(resp, test) for resp, test in zip(responses.responses, tests)
    ]

    passed = sum(scores)
    total = len(tests)
    score = float(passed) / float(total)

    # Print results
    if verbose:
        for resp, test, correct in zip(responses.responses, tests, scores):
            if not correct:
                print("=" * 80)
                print(f"Test input:\n{test['input']}")
                print(f"Expected output:\n{test['output']}")
                print(f"Actual output:\n{extract_response_text(resp)}")
        print("=" * 80)
        print(f"Passed {passed}/{total} tests.")
        print("=" * 80)
        print()

    return score


