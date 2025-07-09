import verifiers as vf
from datasets import load_dataset
from verifiers import Environment


def load_gsm8k_environment(env_args: dict = {}) -> Environment:
    from verifiers.utils.data_utils import extract_hash_answer, extract_boxed_answer

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(
        lambda x: {
            "question": x["question"],
            "answer": extract_hash_answer(x["answer"]),
            "info": {},
            "task": "gsm8k",
        },
        remove_columns=dataset.column_names,  # type: ignore
    )  # type: ignore

    parser = vf.ThinkParser(extract_fn=extract_boxed_answer)  # uses \boxed{...} to parse the answer by default

    def correct_answer_reward_func(completion, answer, **kwargs) -> float:
        response = parser.parse_answer(completion) or ""
        print(response, answer)
        return 1.0 if response == str(answer) else 0.0

    rubric = vf.Rubric(
        funcs=[
            correct_answer_reward_func,
            parser.get_format_reward_func(),
        ],
        weights=[1.0, 0.2],
    )

    system_prompt = """\
Think step by step inside <think>...</think> tags.

Provide the final numerical answer inside \\boxed{{...}}."""

    vf_env = vf.SingleTurnEnv(dataset=dataset, system_prompt=system_prompt, parser=parser, rubric=rubric)
    return vf_env


def load_simple_math_environment(env_args: dict = {}) -> Environment:
    import json

    def get_valid_answers(x: dict) -> list[str]:
        one_or_more_answers = json.loads(x["verification_info"])["ground_truth"]
        if isinstance(one_or_more_answers, str):
            return [one_or_more_answers]
        elif isinstance(one_or_more_answers, list):
            return one_or_more_answers
        else:
            print("Invalid answer format:", one_or_more_answers)
            return []
            # raise ValueError(f"Invalid answer format: {one_or_more_answers}")

    train_dataset = load_dataset("justus27/math-hendrycks-genesys-format", split="train").map(
        lambda x: {"question": x["prompt"], "info": {"answers": get_valid_answers(x)}, "task": "simple-math"}
    )
    train_dataset = train_dataset.remove_columns(["prompt", "verification_info"])

    from verifiers.utils.data_utils import extract_boxed_answer

    parser = vf.ThinkParser()
    from zeroband.training.orchestrator.genesys.math_utils import grade_answer_mathd, grade_answer_sympy

    def grade_answer(response: str, answers: list[str]) -> float:
        for answer in answers:
            if "\\boxed{" in answer:
                answer = extract_boxed_answer(answer)
            if grade_answer_mathd(response, answer) or grade_answer_sympy(response, answer):
                return 1.0
        return 0.0

    def correct_answer_reward_func(completion, info, **kwargs) -> float:
        response = parser.parse_answer(completion) or ""
        if "\\boxed{" in response:
            response = extract_boxed_answer(response)
        is_correct = [grade_answer(response, answer) for answer in info["answers"]]
        return max(is_correct)

    rubric = vf.Rubric(
        funcs=[
            correct_answer_reward_func,
        ],
        weights=[1.0],
    )
    vf_env = vf.SingleTurnEnv(dataset=train_dataset, parser=parser, rubric=rubric)
    return vf_env


def load_reverse_environment(env_args: dict = {}) -> Environment:
    import json

    # import re
    # train_dataset = load_dataset("agentlans/wikipedia-paragraphs", split="train").map(
    #    lambda x: {
    #        "question": "Reverse the text in quotation marks character-by-character: "
    #        + x["text"][:50].strip()
    #        + "\n\nPut your final answer in <answer>...</answer> tags.",
    #        "answer": x["text"][:50][::-1].strip(),
    #        "info": {},
    #        "task": "reverse-text",
    #    }
    # )
    train_dataset = load_dataset("mikasenghaas/reverse_text_dataset_debug_50_seq_len", split="train").map(
        lambda x: {
            "question": x["prompt"],
            "answer": json.loads(x["verification_info"])["ground_truth"],
            "info": {},
            "task": x["task_type"],
        }
    )
    train_dataset = train_dataset.remove_columns(["prompt", "verification_info", "task_type"])

    parser = vf.XMLParser(["answer"], answer_field="answer")

    def lcs_reward_func(completion, answer, **kwargs) -> float:
        """
        LCS ratio of the reversed prompt and the parsed completion.
        """

        def lcs_ratio(x: str, y: str) -> float:
            """
            Return the longest common subsequence ratio of x and y.
            """
            from difflib import SequenceMatcher

            return SequenceMatcher(None, x, y).ratio()

        response = parser.parse_answer(completion) or ""
        return lcs_ratio(response, answer)

    rubric = vf.Rubric(
        funcs=[
            lcs_reward_func,
        ],
        weights=[1.0],
    )

    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        parser=parser,
        rubric=rubric,
    )
    return vf_env


REGISTRY = {
    "gsm8k": load_gsm8k_environment,
    "reverse-text": load_reverse_environment,
    "simple-math": load_simple_math_environment,
}


def get_environment(env_id: str, env_args: dict = {}) -> Environment:
    if env_id not in REGISTRY:
        raise ValueError(f"Environment {env_id} not found")
    return REGISTRY[env_id](env_args)
