import re
import random
from typing import Tuple, Dict, Any, Union

def extract_thinking(solution_str):
    """Extract the thinking process"""
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_answer_from_response(solution_str):
    """
    Extract the answer from the solution string.
    Compatible with both plain answers in <answer> tags and "Answer: ..." format inside <answer> tags.
    """
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None

    # Match <answer> tags
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match:
        answer_content = match.group(1).strip()
        
        # Check if answer content has "Answer: " format
        answer_label_pattern = r'Answer:\s*(.*?)$'
        label_match = re.search(answer_label_pattern, answer_content, re.DOTALL)
        
        if label_match:
            # If "Answer: " format exists, extract that part
            answer = label_match.group(1).strip()
        else:
            # Otherwise use the entire <answer> tag content
            answer = answer_content
        
        answer = re.sub(r'\s+', ' ', answer)
        return answer
    
    # If no <answer> tag found
    return None

def normalize_answer(answer: str) -> Tuple[float, str, bool]:
    """
    Normalize answer format, identify value type and standardize format
    
    Returns:
        (numeric_value, unit/type, is_percentage)
    """
    if answer is None:
        return None, "", False
        
    original = answer
    answer = answer.strip().lower()
    
    # Detect units and format
    is_percentage = '%' in answer
    has_dollar = '$' in answer
    has_million = 'million' in answer.lower()
    has_billion = 'billion' in answer.lower()
    
    # Extract numeric part
    num_pattern = r'[-+]?[0-9]*\.?[0-9]+'
    num_match = re.search(num_pattern, answer)
    
    if not num_match:
        return None, original, False
    
    try:
        # Extract and convert number
        num_value = float(num_match.group(0))
        
        # Prepare return value
        if is_percentage:
            return num_value, '%', True
        elif has_million:
            return num_value, 'million', False
        elif has_billion:
            return num_value, 'billion', False
        elif has_dollar:
            return num_value, '$', False
        else:
            return num_value, '', False
    except:
        return None, original, False

def check_answer_correctness(model_answer: str, expected_answer: str) -> Tuple[bool, str]:
    """Check if model answer is correct, supporting various numeric format comparisons"""
    # Extract model's answer
    extracted_answer = extract_answer_from_response(model_answer)
    
    # If no answer extracted or extracted answer is empty, return early
    if extracted_answer is None:
        return False, ""
    
    # Normalize and extract numeric values
    model_value, model_unit, model_is_percent = normalize_answer(extracted_answer)
    expected_value, expected_unit, expected_is_percent = normalize_answer(expected_answer)
    
    if model_value is None or expected_value is None:
        # If numbers cannot be extracted, fall back to exact string matching
        return extracted_answer.strip().lower() == expected_answer.strip().lower(), extracted_answer
    
    # Check if values and units are equivalent (considering unit variations)
    # For identical values, don't require exactly matching units
    if abs(model_value - expected_value) < 0.01:
        # For very close values, consider correct
        return True, extracted_answer
        
    # Unit conversion and comparison
    is_same_unit_type = False
    
    # Check unit type consistency
    if model_is_percent == expected_is_percent:
        is_same_unit_type = True
    # Special handling for "$X" and "X million" cases
    elif model_unit == '$' and expected_unit == 'million':
        model_value = model_value * 1000000
        is_same_unit_type = True
    elif expected_unit == '$' and model_unit == 'million':
        expected_value = expected_value * 1000000
        is_same_unit_type = True
    # Special handling for "$X" and "X billion" cases
    elif model_unit == '$' and expected_unit == 'billion':
        model_value = model_value * 1000000000
        is_same_unit_type = True
    elif expected_unit == '$' and model_unit == 'billion':
        expected_value = expected_value * 1000000000
        is_same_unit_type = True
    # Special handling for "$X million" and "X" cases (assuming X alone means millions)
    elif model_unit == '$' and model_unit == 'million' and expected_unit == '':
        is_same_unit_type = True
    elif expected_unit == '$' and expected_unit == 'million' and model_unit == '':
        is_same_unit_type = True
    
    if not is_same_unit_type:
        # Unit types inconsistent, but if values are identical, still consider a match
        if abs(model_value - expected_value) < 0.01:
            return True, extracted_answer
        return False, extracted_answer
    
    # Allow percentage precision errors (round to 1 decimal place)
    if model_is_percent and expected_is_percent:
        # Round to 1 decimal place before comparing
        model_rounded = round(model_value, 1)
        expected_rounded = round(expected_value, 1)
        return abs(model_rounded - expected_rounded) < 0.15, extracted_answer
    
    # Handle monetary amounts
    if (model_unit in ['$', 'million', 'billion'] or 
        expected_unit in ['$', 'million', 'billion']):
        # Compare after converting to the same unit
        if abs(model_value - expected_value) / max(abs(expected_value), 1) < 0.05:
            return True, extracted_answer
    else:
        # General number comparison, allow small error margin
        relative_error = abs(model_value - expected_value) / max(abs(expected_value), 1)
        if relative_error < 0.05:  # 5% relative error margin
            return True, extracted_answer
    
    return False, extracted_answer

def check_format(solution_str):
    """Check if the solution follows the required format with proper <think> and <answer> tags"""
    thinking = extract_thinking(solution_str)
    answer = extract_answer_from_response(solution_str)
    
    if thinking is None or answer is None:
        return False
    
    return True

def compute_score(solution_str, ground_truth, extra_info=None, format_score=0.1, score=1.0, return_details=False):
    """
    Scoring function for FinQA task - Evaluates only answer accuracy
    
    Args:
        solution_str: The solution text from the model
        ground_truth: String containing the correct answer
        extra_info: Additional information (optional)
        format_score: Score for correct format but wrong answer
        score: Full score for correct answer
        return_details: Whether to return detailed metrics
    """
    # Randomly print debug info (to avoid excessive logs)
    do_print = random.randint(1, 32) == 1
    
    # Handle ground_truth format
    if isinstance(ground_truth, dict) and "ground_truth" in ground_truth:
        correct_answer = ground_truth["ground_truth"]
    else:
        correct_answer = ground_truth
    
    # Extract answer for logging
    predicted_answer = extract_answer_from_response(solution_str)

    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {correct_answer}")
        print(f"Extracted answer: {predicted_answer}")
        print(f"Solution string: {solution_str}")
    
    # Check if format is correct
    format_correct = check_format(solution_str)
    
    # If format is incorrect, return zero score
    if not format_correct:
        if do_print:
            print("Incorrect format: missing proper <think> or <answer> tags")
        
        if return_details:
            return {
                'combined_score': 0.0,
                'accuracy': 0.0,
                'format_correct': format_correct
            }
        return 0.0
    
    # Check if answer is correct
    is_correct, extracted_answer = check_answer_correctness(solution_str, correct_answer)
    
    # Calculate accuracy
    accuracy = 1.0 if is_correct else 0.0
    
    # Calculate final score
    if is_correct:
        if do_print:
            print(f"Correct answer: {extracted_answer}")
        final_score = score
    else:
        if do_print:
            print(f"Format correct but answer incorrect: {extracted_answer}, Expected: {correct_answer}")
        final_score = format_score
    
    # Return detailed metrics if requested
    if return_details:
        return {
            'combined_score': final_score,
            'accuracy': accuracy,
            'format_correct': format_correct
        }
    
    return final_score

