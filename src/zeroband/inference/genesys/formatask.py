import difflib
import re
from typing import Dict


def compute_reward(
    completion: str, 
    verification_info: Dict,
    tag_name: str = "extracted_formatted"
):
    """
    Generic difflib-based reward computation for tasks expecting extracted content in XML tags.
    
    Args:
        completion: The model's completion text
        verification_info: Dictionary containing ground truth
        tag_name: XML tag name to extract content from
        
    Returns:
        Float reward between 0 and 1
    """
    # Extract answer from specified tag
    tag_pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    if f"<{tag_name}>" not in completion:
        return 0
    
    answer_match = re.search(tag_pattern, completion, re.DOTALL)
    if not answer_match:
        return 0

    # Get ground truth
    ground_truth = verification_info.get("ground_truth")
    if not ground_truth:
        return 0

    try:
        # Clean and split both into lines
        answer_lines = answer_match.group(1).strip().split("\n")
        truth_lines = ground_truth.strip().split("\n")

        # Use difflib to compare line sequences
        matcher = difflib.SequenceMatcher(None, answer_lines, truth_lines)

        # Calculate similarity ratio
        reward = matcher.ratio()

        return reward

    except Exception:
        return 0