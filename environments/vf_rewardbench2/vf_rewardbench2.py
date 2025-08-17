import difflib
from typing import Any, Dict, List

import verifiers as vf
from datasets import load_dataset


def _to_text(candidate: Any) -> str:
	"""Best-effort conversion of candidate (str | list[dict] | dict) to text."""
	if candidate is None:
		return ""
	# Already a string
	if isinstance(candidate, str):
		return candidate
	# List of messages or strings
	if isinstance(candidate, list):
		# If list of message dicts
		if candidate and isinstance(candidate[0], dict) and "content" in candidate[0]:
			return "\n".join(str(m.get("content", "")) for m in candidate)
		# If list of strings
		return "\n".join(str(x) for x in candidate)
	# Single message dict
	if isinstance(candidate, dict) and "content" in candidate:
		return str(candidate.get("content", ""))
	# Fallback
	return str(candidate)


def _extract_prompt_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
	"""Extract prompt messages in verifiers chat format [{role, content}]."""
	# Common RewardBench-style schemas
	for key in ("prompt_messages", "messages"):
		if key in example and isinstance(example[key], list):
			msgs = example[key]
			if msgs and isinstance(msgs[0], dict) and "role" in msgs[0] and "content" in msgs[0]:
				# Use as-is
				return [
					{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
					for m in msgs
				]
	# Single text prompt
	for key in ("prompt", "instruction", "question"):
		if key in example and isinstance(example[key], str):
			return [{"role": "user", "content": example[key]}]
	# Fallback: empty user message
	return [{"role": "user", "content": ""}]


def load_environment(
	split: str | None = None,
	include_categories: List[str] | None = None,
	**kwargs,
) -> vf.Environment:
	"""
	Load RewardBench2 as a SingleTurnEnv with a preference-style reward.

	Args:
		split: HF split to load. If None, will try 'train' then 'validation' then 'test'.
		include_categories: Optional list of category names to keep if dataset has a 'category' column.
	"""
	# Resolve split with graceful fallback
	splits_to_try: List[str] = []
	if split is not None:
		splits_to_try = [split]
	else:
		splits_to_try = ["train", "validation", "test"]

	ds = None
	last_err = None
	for sp in splits_to_try:
		try:
			ds = load_dataset("allenai/reward-bench-2", split=sp)
			break
		except Exception as e:  # noqa: BLE001
			last_err = e
	if ds is None:
		raise RuntimeError(f"Failed to load allenai/reward-bench-2 (tried splits {splits_to_try}). Last error: {last_err}")

	# Optional category filtering if available
	if include_categories and "category" in ds.column_names:
		allowed = set(include_categories)
		ds = ds.filter(lambda x: x.get("category") in allowed)

	# Map dataset to verifiers-friendly fields
	def process_example(x: Dict[str, Any]) -> Dict[str, Any]:
		prompt_messages = _extract_prompt_messages(x)
		chosen = _to_text(x.get("chosen") or x.get("preferred") or x.get("chosen_response"))
		rejected = _to_text(x.get("rejected") or x.get("dispreferred") or x.get("rejected_response"))
		return {
			"prompt": prompt_messages,
			"info": {
				"chosen": chosen,
				"rejected": rejected,
				"category": x.get("category"),
				"source": x.get("source") or x.get("dataset"),
			},
			"task": "rewardbench2",
		}

	dataset = ds.map(process_example)

	# Keep only needed columns if present
	keep_cols = ["prompt", "info", "task"]
	cols_to_remove = [c for c in dataset.column_names if c not in keep_cols]
	if cols_to_remove:
		dataset = dataset.remove_columns(cols_to_remove)

	def preference_reward(completion, info: Dict[str, Any], **k) -> float:
		# Extract last assistant message
		try:
			completion_text = completion[-1]["content"]
		except Exception:  # noqa: BLE001
			completion_text = _to_text(completion)

		chosen = _to_text((info or {}).get("chosen"))
		rejected = _to_text((info or {}).get("rejected"))

		def sim(a: str, b: str) -> float:
			if not a or not b:
				return 0.0
			return difflib.SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()

		sim_chosen = sim(completion_text, chosen)
		sim_rejected = sim(completion_text, rejected)

		# Map difference to [0, 1]
		score = 0.5 * (sim_chosen - sim_rejected) + 0.5
		if score < 0.0:
			return 0.0
		if score > 1.0:
			return 1.0
		return score

	rubric = vf.Rubric(
		funcs=[
			preference_reward,
		],
		weights=[1.0],
	)

	vf_env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
	return vf_env