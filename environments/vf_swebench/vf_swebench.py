import difflib
import json
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset


def load_environment(
    hf_dataset_path: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "train",
    limit: int | None = None,
    max_concurrent: int = 8,
    **kwargs: Any,
) -> vf.Environment:
    """
    SWE-bench style environment.

    The model is prompted with a bug description and should return a unified diff patch inside
    <patch>...</patch> XML tags (optionally preceded by <think>...</think> thought).

    Reward combines:
      - format/structure checks for unified diff-like output
      - text similarity to the ground-truth patch when available
    """

    def build_prompt(example: dict) -> str:
        repo = example.get("repo") or example.get("repository") or "<unknown-repo>"
        base = example.get("base_commit") or example.get("commit") or "<unknown-commit>"
        problem = (
            example.get("problem_statement")
            or example.get("prompt")
            or example.get("issue")
            or example.get("task")
            or "Fix the bug described below and return a unified diff patch."
        )
        hint = example.get("hints") or example.get("hints_text")

        # Light, self-contained instruction that matches our XMLParser usage
        instruction = [
            "You are given a repository and base commit.",
            f"Repository: {repo}",
            f"Base commit: {base}",
            "Problem:",
            str(problem).strip(),
        ]
        if hint:
            instruction += ["Hints:", str(hint).strip()]

        instruction += [
            "Produce only a patch in unified diff format inside <patch>...</patch>.",
            "You may optionally reason in <think>...</think> before the patch.",
            "Do not include commands to run. Only the patch content.",
        ]
        return "\n".join(instruction)

    def load_or_build_dataset() -> Dataset:
        try:
            ds = load_dataset(hf_dataset_path, split=split)
        except Exception:
            # Fall back to a tiny in-memory demo if HF dataset is unavailable
            data = [
                {
                    "repo": "example/project",
                    "base_commit": "deadbeef",
                    "problem_statement": "A function returns an incorrect sum when inputs are negative.",
                    "patch": """diff --git a/math.py b/math.py
index e69de29..b1f2c00 100644
--- a/math.py
+++ b/math.py
@@ -1,4 +1,4 @@
-def add(a, b):
-    return a - b
+def add(a, b):
+    return a + b
""",
                }
            ]
            return Dataset.from_list(data)

        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        return ds

    raw_ds = load_or_build_dataset()

    def to_env_example(x: dict) -> dict:
        # Ground-truth patch field varies across SWE-bench variants; try common keys
        gt_patch = (
            x.get("patch")
            or x.get("diff")
            or x.get("ground_truth_patch")
            or x.get("solution_patch")
            or ""
        )
        return {
            "prompt": [{"role": "user", "content": build_prompt(x)}],
            "answer": gt_patch,
            "task": "swebench",
            "info": {
                k: x.get(k)
                for k in (
                    "repo",
                    "repository",
                    "base_commit",
                    "commit",
                    "instance_id",
                    "tests",
                    "test_commands",
                )
                if k in x
            },
        }

    dataset = raw_ds.map(to_env_example)

    parser = vf.XMLParser(["think", "patch"], answer_field="patch")

    def patch_structure_reward(completion, **kwargs) -> float:
        parsed = parser.parse_answer(completion)
        if not parsed:
            return 0.0
        try:
            text = parsed.strip()
        except Exception:
            return 0.0

        score = 1.0
        # Basic unified diff markers
        has_file_headers = ("--- " in text) and ("+++ " in text)
        has_hunks = "@@" in text
        has_git_header = "diff --git" in text or text.startswith("Index: ")

        if not has_file_headers:
            score *= 0.5
        if not has_hunks:
            score *= 0.7
        if not has_git_header:
            score *= 0.8

        # Penalize extremely short patches
        if len(text.splitlines()) < 4:
            score *= 0.5

        return float(max(0.0, min(1.0, score)))

    def patch_similarity_reward(completion, answer: str, **kwargs) -> float:
        parsed = parser.parse_answer(completion)
        if not parsed:
            return 0.0
        if not isinstance(answer, str) or not answer:
            # If we don't have a ground truth patch, fall back to structure only
            return patch_structure_reward(completion)
        try:
            matcher = difflib.SequenceMatcher(None, parsed, answer)
            return float(matcher.ratio())
        except Exception:
            return 0.0

    rubric = vf.Rubric(
        funcs=[
            patch_structure_reward,
            patch_similarity_reward,
        ],
        weights=[0.3, 0.7],
    )

    return vf.SingleTurnEnv(dataset=dataset, parser=parser, rubric=rubric, max_concurrent=max_concurrent)