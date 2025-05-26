import re
import string
import random
import collections
from typing import Dict, Any, Tuple, Union

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

def clean_answer_tags(answer_text):
    """Clean answer text and handle various answer formats"""
    if not answer_text:
        return ""
    
    # Remove <answer> and </answer> tags
    cleaned_text = re.sub(r'</?answer>', '', answer_text)
    
    # Check if it contains a thinking block
    think_pattern = r'<think>(.*?)</think>\s*'
    think_match = re.search(think_pattern, cleaned_text, re.DOTALL)
    
    # If thinking block found
    if think_match:
        # Get the thinking content
        think_content = think_match.group(1)
        # Extract content after the thinking block
        after_think_content = cleaned_text[think_match.end():].strip()
        
        # If there's content after the thinking block, use it
        if after_think_content:
            cleaned_text = after_think_content
        else:
            # If no content after thinking block, search for answer within the thinking block
            answer_pattern = r'(?:the\s+answer\s+is\s*:?\s*)(.*?)(?:$|\.|\n)'
            answer_match = re.search(answer_pattern, think_content, re.IGNORECASE)
            
            if answer_match:
                # Extract answer from thinking block
                extracted_answer = answer_match.group(1).strip()
                # Handle potential quotes and periods
                extracted_answer = re.sub(r'^["\'](.*?)["\']\.?$', r'\1', extracted_answer)
                return extracted_answer
            
            # If no explicit "the answer is" in thinking block, try matching "Therefore, ..." or "Thus, ..."
            conclusion_pattern = r'(?:therefore|thus|so|hence|consequently|in conclusion)[,:]?\s+(.*?)(?:$|\.|\n)'
            conclusion_match = re.search(conclusion_pattern, think_content, re.IGNORECASE)
            
            if conclusion_match:
                return conclusion_match.group(1).strip()
    
    # Look for "the answer is" pattern in the complete text
    answer_pattern = r'(?:the\s+answer\s+is\s*:?\s*)(.*?)(?:$|\.|\n)'
    answer_match = re.search(answer_pattern, cleaned_text, re.IGNORECASE)
    
    if answer_match:
        # Extract matched content
        extracted_answer = answer_match.group(1).strip()
        # Handle potential quotes
        extracted_answer = re.sub(r'^["\'](.*)["\']$', r'\1', extracted_answer)
        return extracted_answer
    
    # If no specific pattern found, return the cleaned text
    return cleaned_text.strip()

def normalize_answer(s):
    """Normalize answer text: lowercase, remove punctuation, articles and extra spaces"""
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
    """Calculate exact match score"""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    """Calculate F1 score"""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If answer is empty, F1 is 1 (if both are empty), otherwise 0
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def check_format(solution_str):
    """Check if the answer follows the required format with <think> and <answer> tags"""
    thinking = extract_thinking(solution_str)
    answer = extract_answer(solution_str)
    
    if thinking is None or answer is None or not answer:
        return False
    
    return True

def compute_score(solution_str, ground_truth, extra_info=None, format_score=0.1, score=1.0, return_details=False):
    """
    Scoring function for OTTQA task - Evaluates answer accuracy and ensures correct format
    
    Args:
        solution_str: The solution text from the model
        ground_truth: String containing the correct answer or dictionary
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
    
    # Check if format is correct
    format_correct = check_format(solution_str)
    
    # If format is incorrect, return zero score
    if not format_correct:
        if do_print:
            print("Incorrect format: missing proper <think> or <answer> tags")
        
        if return_details:
            return {
                'combined_score': 0.0,
                'em': 0.0,
                'f1': 0.0,
                'format_correct': format_correct
            }
        return 0.0
    
    # Extract and clean answer
    extracted_answer = extract_answer(solution_str)
    cleaned_answer = clean_answer_tags(extracted_answer)
    
    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {correct_answer}")
        print(f"Extracted answer: {cleaned_answer}")
        print(f"Solution string: {solution_str}")
    
    # Calculate evaluation metrics
    exact_match = compute_exact(correct_answer, cleaned_answer)
    f1_score = compute_f1(correct_answer, cleaned_answer)
    
    # Use F1 score and exact match to determine if the answer is correct
    # Usually use F1 score as the main metric, but if exact match is 1, the answer is completely correct
    is_correct = exact_match == 1 or f1_score > 0.85  # F1 score threshold can be adjusted
    
    # Calculate final score
    if is_correct:
        if do_print:
            print(f"Correct answer: '{cleaned_answer}', Expected: '{correct_answer}'")
            print(f"EM: {exact_match}, F1: {f1_score}")
        final_score = score
    else:
        if do_print:
            print(f"Format correct but answer incorrect: '{cleaned_answer}', Expected: '{correct_answer}'")
            print(f"EM: {exact_match}, F1: {f1_score}")
        # If F1 score is high but below threshold, provide partial reward
        if f1_score > 0.5:
            # Adjust format score based on F1 score, up to 80% of full score
            adjusted_score = format_score + (score - format_score) * (f1_score - 0.5) * 2
            final_score = min(adjusted_score, score * 0.8)
        else:
            final_score = format_score
    
    # Return detailed metrics if requested
    if return_details:
        return {
            'combined_score': final_score,
            'em': exact_match,
            'f1': f1_score,
            'format_correct': format_correct
        }
    
    return final_score
