import re
import random
import json
import string
from typing import Dict, Any, Union

    
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

def normalize_answer(s):
    """Normalize answer text: convert to lowercase, remove punctuation, articles and extra spaces"""
    if not s:
        return ""
        
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    """Get normalized tokens from text"""
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    """Compute exact match score"""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    """Compute F1 score to evaluate partial match quality"""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is empty, F1 is 0
        return 0
    
    # Calculate common tokens
    common = set(gold_toks) & set(pred_toks)
    
    # If no common tokens, F1 is 0
    if len(common) == 0:
        return 0
    
    # Calculate precision and recall
    precision = len(common) / len(pred_toks)
    recall = len(common) / len(gold_toks)
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1

def check_format(solution_str):
    """Check if the solution follows the required format with proper <think> and <answer> tags."""
    thinking = extract_thinking(solution_str)
    answer = extract_answer(solution_str)
    
    if thinking is None or answer is None:
        return False, 0
        
    if len(answer.strip()) < 1: 
        return False, 0
    
    # Reward for longer thinking process
    thinking_len = len(thinking.strip())
    if thinking_len >= 500:
        thinking_score = 0.05
    else:
        thinking_score = 0 
    
    return True, thinking_score

def compute_score(solution_str, ground_truth, extra_info=None, method='strict', format_score=0.1, score=1.0, return_details=False):
    """
    The scoring function for HybridQA task.
    
    Args:
        solution_str: the solution text from the model
        ground_truth: string or dictionary containing the correct answer
        extra_info: additional information (not used in current implementation)
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
        return_details: whether to return detailed metrics including EM and F1
    """
    # Extract predicted answer
    predicted_answer = extract_answer(solution_str=solution_str)
    
    # For debugging (print randomly to avoid excessive logs)
    do_print = random.randint(1, 64) == 1
    
    # Handle ground_truth possible formats (string or dictionary)
    if isinstance(ground_truth, dict) and "ground_truth" in ground_truth:
        correct_answer = ground_truth["ground_truth"]
    else:
        correct_answer = ground_truth
    
    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {correct_answer}")
        print(f"Extracted answer: {predicted_answer}")
        print(f"Solution string snippet: {solution_str}")

    # First check if format is correct and get thinking score
    format_correct, thinking_score = check_format(solution_str)

    # Initialize EM and F1 metrics
    em_score = 0  # Default to 0 indicating incorrect
    f1_score = 0  # Default to 0 indicating no match

    if not format_correct:
        if do_print:
            print("Incorrect format: missing proper <think> or <answer> tags")
            
        if return_details:
            return {
                'combined_score': 0.0,
                'em': em_score,
                'f1': f1_score,
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
                'em': em_score,
                'f1': f1_score,
                'format_correct': format_correct
            }
        return thinking_score 
    
    # Handle multiple possible correct answers (separated by |)
    if isinstance(correct_answer, str) and "|" in correct_answer:
        possible_answers = [ans.strip() for ans in correct_answer.split("|")]
        # Check each possible answer
        exact_matches = [compute_exact(ans, predicted_answer) for ans in possible_answers]
        f1_scores = [compute_f1(ans, predicted_answer) for ans in possible_answers]
        
        # Take the highest scores
        best_exact = max(exact_matches) if exact_matches else 0
        best_f1 = max(f1_scores) if f1_scores else 0
        
        is_correct = best_exact == 1
        em_score = best_exact  # Set EM metric
        f1_score = best_f1    # Set F1 metric
    else:
        # Single answer case
        em_score = compute_exact(correct_answer, predicted_answer)
        f1_score = compute_f1(correct_answer, predicted_answer)
        is_correct = em_score == 1

    # Calculate final score
    final_score = 0.0
    if is_correct:
        if do_print:
            print("Correct answer")
        final_score = score
    else:
        # Give partial score based on F1 score
        partial_score = format_score + thinking_score
        
        # Give extra reward for high F1 scores
        if f1_score >= 0.7:
            partial_score += 0.3
        elif f1_score >= 0.5:
            partial_score += 0.2
        elif f1_score >= 0.3:
            partial_score += 0.1
            
        if do_print:
            print(f"Partial match with F1 score {f1_score:.2f}, partial score: {partial_score:.2f}")
        
        # Ensure partial score doesn't exceed maximum
        final_score = min(0.8, partial_score)
    
    # If detailed metrics are requested
    if return_details:
        return {
            'combined_score': final_score,
            'em': em_score,
            'f1': f1_score,
            'format_correct': format_correct
        }
    
    return final_score
