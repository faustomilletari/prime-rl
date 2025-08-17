import re
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset


def load_environment(
    dataset_name: str | None = None,
    dataset_split: str = "train",
    min_word_count_key: str = "min_words",
) -> vf.Environment:
    """
    Load an IFEval-style environment that checks adherence to verifiable instructions.

    Expects dataset fields:
    - question: prompt to the model (or build from prompt)
    - answer: an optional structured answer/constraints, else taken from verification_info
    - info: dict-like verification info (constraints)
    - task: task id string (e.g., "ifeval")

    If dataset_name is None, attempts to load a small demo dataset from in-memory examples.
    """

    def build_default_dataset() -> Dataset:
        # Minimal synthetic set emulating IFEval constraints
        rows: list[dict[str, Any]] = [
            {
                "question": "Write a paragraph about the importance of sleep. Make sure to produce at least 120 words and include the keyword 'circadian' at least twice.",
                "answer": None,
                "info": {
                    "constraints": {
                        "min_words": 120,
                        "keywords": [{"text": "circadian", "min_occurrences": 2}],
                    }
                },
                "task": "ifeval",
            },
            {
                "question": "List 5 bullet points about healthy hydration. Each bullet must start with '- ' and include the word 'water'.",
                "answer": None,
                "info": {
                    "constraints": {
                        "bullet_points": {"count": 5, "prefix": "- "},
                        "keywords": [{"text": "water", "min_occurrences": 5}],
                    }
                },
                "task": "ifeval",
            },
        ]
        return Dataset.from_list(rows)

    if dataset_name:
        ds = load_dataset(dataset_name, split=dataset_split)
        # Heuristically map common fields
        def _map_row(x):
            info = x.get("verification_info") or x.get("info") or {}
            question = x.get("prompt") or x.get("question") or ""
            task = x.get("task") or "ifeval"
            return {
                "question": question,
                "answer": x.get("answer"),
                "info": info if isinstance(info, dict) else {},
                "task": task,
            }
        ds = ds.map(_map_row)
        # Drop unused raw columns if present
        for col in ["prompt", "verification_info"]:
            if col in ds.column_names:
                ds = ds.remove_columns([col])
        dataset = ds
    else:
        dataset = build_default_dataset()

    class IFEvalParser(vf.Parser):
        def parse(self, text: str) -> str:
            return text or ""

        def get_format_reward_func(self):
            # No-op for base formatting; we check constraints in rubric functions
            def f(completion: vf.Messages, **kwargs) -> float:
                return 1.0 if self.parse_answer(completion) is not None else 0.0
            return f

    parser = IFEvalParser()

    # Constraint checkers
    def count_words(text: str) -> int:
        tokens = re.findall(r"\b\w+\b", text)
        return len(tokens)

    def check_min_words(completion: str, constraints: dict, key: str = min_word_count_key) -> float:
        req = constraints.get(key)
        if not isinstance(req, int):
            return 1.0
        return 1.0 if count_words(completion) >= req else 0.0

    def check_keywords(completion: str, constraints: dict) -> float:
        kw = constraints.get("keywords")
        if not isinstance(kw, list):
            return 1.0
        text_lower = completion.lower()
        for item in kw:
            if not isinstance(item, dict):
                continue
            token = str(item.get("text", "")).lower()
            min_occ = int(item.get("min_occurrences", 1))
            if not token:
                continue
            if text_lower.count(token) < min_occ:
                return 0.0
        return 1.0

    def check_bullets(completion: str, constraints: dict) -> float:
        spec = constraints.get("bullet_points")
        if not isinstance(spec, dict):
            return 1.0
        count_req = int(spec.get("count", 0))
        prefix = str(spec.get("prefix", "- "))
        lines = [ln for ln in completion.splitlines() if ln.strip()]
        bullets = [ln for ln in lines if ln.startswith(prefix)]
        if count_req and len(bullets) < count_req:
            return 0.0
        return 1.0

    def ifeval_reward(completion: vf.Messages, info: dict, **kwargs) -> float:
        text = parser.parse_answer(completion) or ""
        constraints = {}
        if isinstance(info, dict):
            # Allow nested info["constraints"] or flat keys
            constraints = info.get("constraints", info)
        # Chain independent checks; all must pass
        checks = [
            check_min_words(text, constraints),
            check_keywords(text, constraints),
            check_bullets(text, constraints),
        ]
        return float(all(score >= 1.0 for score in checks))

    rubric = vf.Rubric(
        funcs=[ifeval_reward],
        weights=[1.0],
    )

    system_prompt = (
        "Follow the user's instructions exactly. When asked to use bullets, start each bullet with '- '."
    )

    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        max_concurrent=10,
    )
    return vf_env