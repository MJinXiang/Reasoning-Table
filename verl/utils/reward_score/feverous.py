import re
import random
import json
from difflib import SequenceMatcher


def extract_answer(solution_str):
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

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match:
        answer_content = match.group(1).strip()
        
        answer_label_pattern = r'Answer:\s*(.*?)$'
        label_match = re.search(answer_label_pattern, answer_content, re.DOTALL)
        
        if label_match:
            answer = label_match.group(1).strip()
        else:
            answer = answer_content
        
        answer = re.sub(r'\s+', ' ', answer)
        return answer
    
    return None
    
def extract_thinking(solution_str):
    """Extract the thinking process from the solution string."""
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None

    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, solution_str, re.DOTALL)  
    if match:
        return match.group(1).strip()
    else:
        return None


def normalize_answer(answer):
    """Normalize the answer string for FEVEROUS task."""
    if answer is None:
        return None
    
    # Convert to lowercase and clean up
    answer = answer.lower().strip()
    answer = re.sub(r'\s+', ' ', answer)
    
    # Standardize the possible answers
    if 'support' in answer:
        return 'SUPPORTS'
    elif 'refute' in answer:
        return 'REFUTES'
    elif 'not enough info' in answer or 'nei' in answer:
        return 'NOT ENOUGH INFO'
    
    # Return the original answer if it doesn't match any of the expected formats
    return answer.upper()


def check_format(solution_str):
    """Check if the solution follows the required format with proper <think> and <answer> tags."""
    thinking = extract_thinking(solution_str)
    answer = extract_answer(solution_str)
    
    if thinking is None or answer is None:
        return False, 0
        
    if len(answer.strip()) < 1: 
        return False, 0
    
    # Reward for detailed thinking
    thinking_len = len(thinking.strip())
    if thinking_len >= 500:
        thinking_score = 0.05
    
    return True, thinking_score


def compute_score(solution_str, ground_truth, extra_info=None, format_score=0.1, score=1.0, return_details=False):
    """
    The scoring function for FEVEROUS task.
    
    Args:
        solution_str: the solution text from the model
        ground_truth: string containing the correct answer (SUPPORTS, REFUTES, NOT ENOUGH INFO)
        extra_info: additional information (not used in current implementation)
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
        return_details: whether to return detailed metrics
    """
    predicted_answer = extract_answer(solution_str=solution_str)
    
    # Handle ground_truth format
    if isinstance(ground_truth, dict) and "ground_truth" in ground_truth:
        correct_answer = ground_truth["ground_truth"]
    else:
        correct_answer = ground_truth

    # For debugging (print randomly to avoid overwhelming logs)
    do_print = random.randint(1, 32) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {correct_answer}")
        print(f"Extracted answer: {predicted_answer}")
        print(f"Solution string: {solution_str}")

    # First check if format is correct and get thinking score
    format_correct, thinking_score = check_format(solution_str)
    
    # Initialize accuracy evaluation
    is_correct = False
    accuracy = 0

    if not format_correct:
        if do_print:
            print("Incorrect format: missing proper <think> or <answer> tags")
            
        if return_details:
            return {
                'combined_score': 0.0,
                'accuracy': 0,
                'format_correct': False
            }
        return 0.0
    
    # Check if answer content is correct
    if predicted_answer is None:
        if do_print:
            print("No answer found in the proper format")
            
        if return_details:
            return {
                'combined_score': thinking_score,
                'accuracy': 0,
                'format_correct': True
            }
        return thinking_score 
    
    # Normalize answers for comparison
    normalized_predicted = normalize_answer(predicted_answer)
    normalized_ground_truth = normalize_answer(correct_answer)
    
    # Check if the prediction is valid
    valid_answers = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    if normalized_predicted not in valid_answers:
        if do_print:
            print(f"Invalid prediction: {normalized_predicted}, expected one of {valid_answers}")
            
        if return_details:
            return {
                'combined_score': thinking_score,
                'accuracy': 0,
                'format_correct': True
            }
        return thinking_score

    # Compare normalized answers
    is_correct = normalized_predicted == normalized_ground_truth
    accuracy = 1 if is_correct else 0

    # Calculate final score
    if is_correct:
        if do_print:
            print(f"Correct answer: {normalized_predicted}")
        final_score = score
    else:
        if do_print:
            print(f"Incorrect answer: predicted '{normalized_predicted}', expected '{normalized_ground_truth}'")
        final_score = format_score + thinking_score
    
    if return_details:
        return {
            'combined_score': final_score,
            'accuracy': accuracy,
            'format_correct': format_correct
        }
    
    return final_score
