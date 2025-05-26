import re
import random
from typing import List, Dict, Any, Tuple
from collections import Counter
import json

def normalize_answer(answer):
    """Normalize the answer for comparison."""
    if answer is None:
        return None
    
    # Convert to lowercase and clean up
    answer = str(answer).lower().strip()
    
    # Remove any brackets or special characters
    answer = re.sub(r'[\[\]\(\){}<>]', '', answer)
    
    # Replace "and" with comma
    answer = re.sub(r'\s+and\s+', ', ', answer)
    
    # Normalize whitespace
    answer = re.sub(r'\s+', ' ', answer)
    
    # Remove percent signs and handle negative values
    answer = answer.replace('%', '')
    
    # Remove dollar signs and other currency symbols
    answer = answer.replace('$', '')
    
    # Remove commas in numbers
    answer = answer.replace(',', '')
    
    return answer


def str_to_num(text):
    """Convert text to number, handling various formats."""
    if text is None:
        return "n/a"
    
    if isinstance(text, (int, float)):
        return float(text)
    
    text = str(text).lower().strip()
    text = text.replace("%", "").replace("$", "").replace(",", "")
    
    # Handle negative numbers with various formats
    text = text.replace("minus ", "-").replace("negative ", "-")
    
    try:
        return float(text)
    except ValueError:
        return "n/a"


def check_answer_correctness(model_answer, expected_answer):
    """
    Check if the model's answer is correct compared to the expected answer.
    Handles various formats including lists, single values, and numerical comparisons.
    """
    
    if model_answer is None:
        return False
    
    # Basic case-insensitive comparison
    if isinstance(model_answer, str) and isinstance(expected_answer, str):
        if model_answer.lower().strip() == expected_answer.lower().strip():
            return True
    
    # Handle single element in list
    if isinstance(expected_answer, list) and len(expected_answer) == 1:
        if isinstance(model_answer, str) and model_answer.lower().strip() == expected_answer[0].lower().strip():
            return True
    
    # Handle string format of list
    if isinstance(expected_answer, str) and expected_answer.startswith("[") and expected_answer.endswith("]"):
        try:
            # Try to parse as list
            import ast
            parsed_answer = ast.literal_eval(expected_answer)
            if isinstance(parsed_answer, list):
                expected_answer = parsed_answer
        except (SyntaxError, ValueError):
            # Remove brackets if parsing fails
            expected_answer = re.sub(r'^\s*\[(.*)\]\s*$', r'\1', expected_answer)
            if ',' in expected_answer:
                expected_answer = [item.strip() for item in expected_answer.split(',')]
    
    # Normalize answers for comparison
    if isinstance(expected_answer, list):
        expected_normalized = [normalize_answer(ans) for ans in expected_answer]
    else:
        expected_normalized = [normalize_answer(expected_answer)]
    
    model_normalized = normalize_answer(model_answer)
    
    # Simple direct comparison (already preprocessed through normalize_answer)
    if model_normalized in expected_normalized:
        return True
    
    # Handle multiple values case (comma separated)
    if ',' in model_normalized:
        model_values = [val.strip() for val in model_normalized.split(',')]
        
        # Try numerical comparison
        model_nums = []
        for val in model_values:
            num = str_to_num(val)
            if num != "n/a":
                model_nums.append(num)
        
        # Multi-value comparison
        if len(model_nums) > 1:
            for exp in expected_normalized:
                if ',' in exp:
                    exp_values = [val.strip() for val in exp.split(',')]
                    exp_nums = []
                    for val in exp_values:
                        num = str_to_num(val)
                        if num != "n/a":
                            exp_nums.append(num)
                    
                    # Compare numerical lists (ignore order)
                    if len(model_nums) == len(exp_nums) and sorted(model_nums) == sorted(exp_nums):
                        return True
        
        # Text multi-value comparison (ignore order)
        for exp in expected_normalized:
            if ',' in exp:
                exp_values = [val.strip() for val in exp.split(',')]
                if len(model_values) == len(exp_values) and sorted(model_values) == sorted(exp_values):
                    return True
    
    # Numerical comparison
    model_num = str_to_num(model_normalized)
    if model_num != "n/a":
        for exp in expected_normalized:
            exp_num = str_to_num(exp)
            if exp_num != "n/a":
                # Check numerical equality (with tolerance)
                if abs(exp_num) < 1e-10:  # Zero value case
                    if abs(model_num) < 1e-5:
                        return True
                else:
                    # Non-zero value, use relative difference
                    relative_diff = abs(model_num - exp_num) / max(abs(exp_num), 1e-10)
                    if relative_diff < 1e-5 or abs(model_num - exp_num) < 1e-5:
                        return True
    
    # Keep only digits comparison
    for exp in expected_normalized:
        exp_digits = re.sub(r'[^0-9.-]', '', exp)
        model_digits = re.sub(r'[^0-9.-]', '', model_normalized)
        
        if exp_digits and model_digits and exp_digits == model_digits:
            return True
    
    return False


def compute_score(predicted_answer, ground_truth, extra_info=None, score=1.0, return_details=False):
    """
    The scoring function for HiTAB task - Exact Answer evaluation only.
    
    Args:
        predicted_answer: the predicted answer from the model
        ground_truth: string containing the correct answer
        extra_info: additional information, may contain key_cells
        score: the score for the correct answer
        return_details: whether to return detailed metrics
    """
    
    # Handle ground_truth format
    if isinstance(ground_truth, dict) and "ground_truth" in ground_truth:
        correct_answer = ground_truth["ground_truth"]
    else:
        correct_answer = ground_truth
        
    # Check if the answer is correct - exact answer evaluation only
    is_correct = check_answer_correctness(predicted_answer, correct_answer)
    
    # Set accuracy (newly added)
    accuracy = 1.0 if is_correct else 0.0

    return {"accuracy": accuracy}