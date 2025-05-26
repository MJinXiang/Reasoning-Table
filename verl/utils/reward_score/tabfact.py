import re
import random
import json

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
    """Normalize the answer string for comparison, converting to numeric format (1/0)."""
    if answer is None:
        return None
    
    # Convert to lowercase and clean up
    answer = answer.lower().strip()
    answer = re.sub(r'\s+', ' ', answer)
    
    # Convert various forms of "supports" to 1
    if 'supports' in answer or 'SUPPORTS' in answer:
        return 1
    # Convert various forms of "refutes" to 0
    elif 'refutes' in answer or 'REFUTES' in answer:
        return 0
    
    # Return None for unrecognized answers
    return None

def check_format(solution_str):
    """Check if the solution follows the required format with proper <think> and <answer> tags."""
    thinking = extract_thinking(solution_str)
    answer = extract_answer(solution_str)
    
    if thinking is None or answer is None:
        return False, 0
        
    if len(answer.strip()) < 1: 
        return False, 0
    
    thinking_len = len(thinking.strip())
    if thinking_len >= 500:
        thinking_score = 0.05  
    else:
        thinking_score = 0 
    
    return True, thinking_score

def compute_score(solution_str, ground_truth, extra_info=None, method='strict', format_score=0.1, score=1.0, return_details=False):
    """
    The scoring function for TabFact task.
    
    Args:
        solution_str: the solution text from the model
        ground_truth: string containing the correct answer (1 for SUPPORTS, 0 for REFUTES)
        extra_info: additional information (not used in current implementation)
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
        return_details: whether to return detailed metrics including accuracy
    """
    predicted_answer = extract_answer(solution_str=solution_str)
    
    # Handle possible JSON string ground_truth
    if isinstance(ground_truth, str):
        try:
            # Try to parse JSON string
            if ground_truth.startswith('"') or ground_truth.startswith("'"):
                # Might be JSON encoded string, like "0" or "1"
                import json
                correct_answer = json.loads(ground_truth)
            else:
                # Might be plain string, like "0" or "1" (without quotes)
                correct_answer = ground_truth
        except:
            # If parsing fails, keep original
            correct_answer = ground_truth
    else:
        correct_answer = ground_truth

    # For debugging (print randomly to avoid overwhelming logs)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {correct_answer}")
        print(f"Extracted answer: {predicted_answer}")
        print(f"Solution string: {solution_str}")

    # Initialize accuracy metric
    accuracy = 0.0  # Default to 0 indicating incorrect
    
    # First check if format is correct and get thinking score
    format_correct, thinking_score = check_format(solution_str)

    if not format_correct:
        if do_print:
            print("Incorrect format: missing proper <think> or <answer> tags")
        
        if return_details:
            return {
                'combined_score': 0.0,
                'accuracy': accuracy,
                'format_correct': format_correct
            }
        return 0.0  
    
    # Check if answer content is correct
    if predicted_answer is None:
        if do_print:
            print("No answer found in the proper format")
            
        if return_details:
            return {
                'combined_score': thinking_score,
                'accuracy': accuracy,
                'format_correct': format_correct
            }
        return thinking_score 
    
    # Normalize and convert answers
    normalized_predicted = normalize_answer(predicted_answer)

    # Convert ground truth to numeric (in case it's a string)
    normalized_ground_truth = int(ground_truth)
    
    # Check for unrecognized answers
    if normalized_predicted is None:
        if do_print:
            print(f"Unrecognized answer format: {predicted_answer}")
            
        if return_details:
            return {
                'combined_score': thinking_score,
                'accuracy': accuracy,
                'format_correct': format_correct
            }
        return thinking_score
    
    # Compare the normalized answers
    is_correct = normalized_predicted == normalized_ground_truth
    
    # Set accuracy metric
    accuracy = 1.0 if is_correct else 0.0

    # Calculate final score
    final_score = 0.0
    if is_correct:
        if do_print:
            print(f"Correct answer - Predicted: {normalized_predicted}, Ground truth: {normalized_ground_truth}")
        final_score = score  
    else:
        if do_print:
            print(f"Correct format but incorrect answer - Predicted: {normalized_predicted}, Ground truth: {normalized_ground_truth}")
        final_score = format_score + thinking_score
    
    # Return detailed metrics if requested
    if return_details:
        return {
            'combined_score': final_score,
            'accuracy': accuracy,
            'format_correct': format_correct
        }
    
    return final_score
