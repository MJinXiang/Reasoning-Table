import re
import random
import string
from collections import Counter
import difflib 
from typing import Dict, Any, Union
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
    """Extract the thinking process from the solution string"""
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

def str_to_num(text):
    """Convert text to number, handle various formats and clean characters"""
    if text is None:
        return "n/a"
    
    if isinstance(text, (int, float)):
        return float(text)
    
    text = str(text)
    text = text.replace("$", "").replace(",", "").replace("_", "")
    
    # Handle percentages
    if "%" in text:
        text = text.replace("%", "")
        try:
            return float(text) / 100
        except ValueError:
            return "n/a"
    
    # Handle regular numbers
    try:
        return float(text)
    except ValueError:
        return "n/a"

def normalize_answer(s):
    """Normalize answer text, remove punctuation and extra spaces"""
    if s is None:
        return ""
    
    # If it's a number, return original format
    if isinstance(s, (int, float)):
        return str(s)
    
    s = str(s).lower()
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'[^\w\s]', '', s)
    return s

def get_tokens(s):
    """Split text into tokens"""
    if not s:
        return []
    return normalize_answer(s).split()

def exact_match_score(prediction, ground_truth):
    """Calculate exact match score"""
    if prediction is None or ground_truth is None:
        return 0
    
    # Handle numeric exact match
    pred_num = str_to_num(prediction)
    truth_num = str_to_num(ground_truth)
    
    if pred_num != "n/a" and truth_num != "n/a":
        # Use relative tolerance for comparison
        if isinstance(pred_num, (int, float)) and isinstance(truth_num, (int, float)):
            # Handle zero value special case
            if truth_num == 0:
                return 1.0 if abs(pred_num) < 1e-5 else 0.0
            
            # Use relative error
            relative_diff = abs(pred_num - truth_num) / max(abs(truth_num), 1e-10)
            return 1.0 if relative_diff < 1e-5 else 0.0
    
    # Handle text exact match
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

def compute_f1(prediction, ground_truth):
    """Calculate F1 score"""
    if prediction is None or ground_truth is None:
        return 0.0
    
    # First try numerical comparison
    pred_num = str_to_num(prediction)
    truth_num = str_to_num(ground_truth)
    
    if pred_num != "n/a" and truth_num != "n/a":
        # If both are numerical, use exact match as F1
        if abs(truth_num) < 1e-10:
            return 1.0 if abs(pred_num) < 1e-5 else 0.0
        
        rel_tol = min(abs(truth_num) * 0.01, 0.1)
        abs_tol = min(abs(truth_num) / 1000, 0.1)
        
        return 1.0 if abs(pred_num - truth_num) <= max(rel_tol, abs_tol) else 0.0
    
    # Text F1 calculation
    prediction_tokens = get_tokens(prediction)
    ground_truth_tokens = get_tokens(ground_truth)
    
    # If both are empty, return F1=1.0
    if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
        return 1.0
    
    # Calculate common tokens
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    # If no common tokens, return F1=0
    if num_same == 0:
        return 0.0
    
    # Calculate precision and recall
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    
    # Calculate F1
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def check_format(solution_str):
    """Check if the solution follows the required format, with appropriate <think> and <answer> tags"""
    thinking = extract_thinking(solution_str)
    answer = extract_answer(solution_str)
    
    if thinking is None or answer is None:
        return False, 0
        
    if len(answer.strip()) < 1: 
        return False, 0
    
    # Reward based on thinking process detail level
    thinking_len = len(thinking.strip())
    if thinking_len >= 500:
        thinking_score = 0.05 
    else:
        thinking_score = 0 
    
    return True, thinking_score

def compute_score(solution_str, ground_truth, extra_info=None, format_score=0.1, max_score=1.0, return_details=False):
    """
    Scoring function for MultiHierTT task.
    
    Args:
        solution_str: The model's solution text
        ground_truth: String or dictionary containing the correct answer
        extra_info: Additional information (not used in current implementation)
        format_score: Score for correct format but wrong answer
        max_score: Maximum score for correct answer
        return_details: Whether to return detailed metrics including EM and F1
    """
    # Extract predicted answer
    predicted_answer = extract_answer(solution_str=solution_str)
    
    # Randomly print some results for debugging
    do_print = random.randint(1, 32) == 1

    # Enhanced ground_truth handling logic
    if isinstance(ground_truth, dict) and "ground_truth" in ground_truth:
        correct_answer = ground_truth["ground_truth"]
    else:
        # Check if it's a JSON string
        if isinstance(ground_truth, str):
            try:
                if (ground_truth.startswith('"') and ground_truth.endswith('"')) or \
                   (ground_truth.startswith('[') and ground_truth.endswith(']')) or \
                   (ground_truth.startswith('{') and ground_truth.endswith('}')):
                    parsed = json.loads(ground_truth)
                    
                    # If parsed result is dict and contains ground_truth
                    if isinstance(parsed, dict) and "ground_truth" in parsed:
                        correct_answer = parsed["ground_truth"]
                    else:
                        correct_answer = parsed
                else:
                    correct_answer = ground_truth
            except:
                # Use original value if parsing fails
                correct_answer = ground_truth
        else:
            correct_answer = ground_truth
    
    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {correct_answer}")
        print(f"Extracted answer: {predicted_answer}")
        print(f"Solution snippet: {solution_str}")

    # Initialize EM and F1 metrics
    em_score = 0.0
    f1_score = 0.0
    
    # Check if format is correct and get thinking score
    format_correct, thinking_score = check_format(solution_str)

    if not format_correct:
        if do_print:
            print("Incorrect format: missing <think> or <answer> tags")
        
        if return_details:
            return {
                'combined_score': 0.0,
                'em': em_score,
                'f1': f1_score,
                'format_correct': format_correct
            }
        return 0.0  
    
    if predicted_answer is None:
        if do_print:
            print("No properly formatted answer found")
        
        if return_details:
            return {
                'combined_score': thinking_score,
                'em': em_score,
                'f1': f1_score,
                'format_correct': format_correct
            }
        return thinking_score 
    
    # Handle multiple possible correct answers (separated by |)
    if isinstance(correct_answer, str) and "|" in correct_answer:
        possible_answers = [ans.strip() for ans in correct_answer.split("|")]
        exact_matches = [exact_match_score(predicted_answer, ans) for ans in possible_answers]
        f1_scores = [compute_f1(predicted_answer, ans) for ans in possible_answers]
        
        best_exact = max(exact_matches) if exact_matches else 0
        best_f1 = max(f1_scores) if f1_scores else 0
        
        # Consider F1=1.0 as correct answer as well
        is_correct = best_exact == 1 or best_f1 == 1.0
        
        # Set EM and F1 metric values
        em_score = best_exact
        f1_score = best_f1
    else:
        # Single answer case
        em_score = exact_match_score(predicted_answer, correct_answer)
        f1_score = compute_f1(predicted_answer, correct_answer)
        
        # Consider F1=1.0 as correct answer as well
        is_correct = em_score == 1.0 or f1_score == 1.0

    final_score = 0.0
    
    # Calculate final score without evidence weighting
    if is_correct:
        if do_print:
            print("Answer is correct")
        final_score = max_score
    else:
        # Give partial score based on F1 evaluation
        if f1_score > 0:
            # Calculate partial score
            partial_score = format_score + thinking_score
            
            if f1_score >= 0.8:
                partial_score += 0.5
            elif f1_score >= 0.6:
                partial_score += 0.3
            elif f1_score >= 0.4:
                partial_score += 0.2
            elif f1_score >= 0.2:
                partial_score += 0.1
            
            if do_print:
                print(f"Partial match, F1 score: {f1_score:.2f}, Total score: {partial_score:.2f}")
            
            # Ensure partial score doesn't exceed maximum
            final_score = min(0.9, partial_score)
        else:
            if do_print:
                print(f"Format correct but answer wrong, thinking score: {thinking_score}")
            final_score = format_score + thinking_score
    
    # Return detailed metrics if requested
    if return_details:
        return {
            'combined_score': final_score,
            'em': em_score,
            'f1': f1_score,
            'format_correct': format_correct
        }
    
    return final_score

