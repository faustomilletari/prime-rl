import verifiers as vf
from datasets import load_dataset
from verifiers import Environment
from verifiers.utils import load_example_dataset


def load_gsm8k_environment(env_args: dict | None = None) -> Environment:
    dataset = load_example_dataset("gsm8k", split="train")
    parser = vf.ThinkParser()  # uses \boxed{...} to parse the answer

    def correct_answer_reward_func(completion, answer, **kwargs) -> float:
        response = parser.parse_answer(completion) or ""
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


def load_reverse_environment(env_args: dict | None = None) -> Environment:
    train_dataset = load_dataset("agentlans/wikipedia-paragraphs", split="train").map(
        lambda x: {"question": x["text"], "answer": x["text"][::-1]}
    )
    parser = vf.XMLParser(["think", "answer"], answer_field="answer")
    system_prompt = f"""Reverse the given text.

    Respond in the following format:
    {parser.get_format_str()}"""

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
            parser.get_format_reward_func(),
        ],
        weights=[1.0, 0.2],
    )

    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    return vf_env


def load_default_environment(env_args: dict = {}) -> Environment:
    dataset_name = env_args.get("dataset", "openai/gsm8k")
    dataset = load_dataset(dataset_name, split="train")
    vf_env = vf.SingleTurnEnv(dataset=dataset)
    return vf_env


REGISTRY = {
    "gsm8k": load_gsm8k_environment,
    "reverse-text": load_reverse_environment,
    "default": load_default_environment,
}


def get_environment(env_id: str, env_args: dict = {}) -> Environment:
    if env_id not in REGISTRY:
        raise ValueError(f"Environment {env_id} not found")
    return REGISTRY[env_id](env_args)
