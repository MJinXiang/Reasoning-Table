import re
import random
import json
from typing import Dict, List, Union, Any

# Import TaTQA evaluation tools
from .tatqa_utils.tatqa_metric import TaTQAEmAndF1, normalize_multi_span
from .tatqa_utils.tatqa_utils import normalize_answer, is_number, to_number, scale_to_num


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
    # Handle different formats of model outputs
    if "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    
    # Look for think tag
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, solution_str, re.DOTALL)  
    if match:
        return match.group(1).strip()
    
    return None

def evaluate_prediction(model_answer, ground_truth_dict, scale=None):
    """Evaluate the model answer against the ground truth."""
    # Initialize evaluator
    em_and_f1 = TaTQAEmAndF1()
    
    # Extract fields
    answer_type = ground_truth_dict.get("answer_type", "")
    gold_answer = ground_truth_dict.get("answer", "")
    correct_scale = scale if scale is not None else ground_truth_dict.get("scale", "")
    
    # Store original answer before preprocessing for potential debugging
    original_model_answer = model_answer
    
    # Handle possible JSON strings in ground_truth
    if isinstance(gold_answer, str) and (gold_answer.startswith("[") or gold_answer.startswith("\"") or gold_answer.startswith("{")):
        try:
            gold_answer = json.loads(gold_answer)
        except:
            pass

    # Preprocess model answer based on answer type and standard answer format
    if answer_type == "multi-span" and not isinstance(model_answer, list):
        model_answer = normalize_multi_span(model_answer)
    
    elif answer_type == "count":
        # Special handling for count type
        if isinstance(model_answer, list) and len(model_answer) > 0:
            model_answer = model_answer[0]
        
        # Clean symbols and convert to integer
        try:
            # Remove currency symbols and thousand separators
            clean_answer = re.sub(r'[$€£¥]', '', str(model_answer))
            clean_answer = clean_answer.replace(',', '')
            model_answer = str(int(float(clean_answer)))
        except:
            model_answer = str(model_answer)
                
        # Process gold_answer similarly
        if isinstance(gold_answer, list) and len(gold_answer) > 0:
            gold_answer = gold_answer[0]
        try:
            gold_answer = str(int(float(str(gold_answer))))
        except:
            gold_answer = str(gold_answer)
    
    elif answer_type == "arithmetic":
        # For arithmetic type, check units and format
        
        # If model answer is string, check and handle format
        if isinstance(model_answer, str):
            # Check if it contains unit words
            has_unit = bool(re.search(r'\b(thousand|million|billion|percent|%)\b', model_answer, re.IGNORECASE))
            
            # If answer contains unit words and scale also specifies units, extract numeric part
            if has_unit and correct_scale:
                # Extract numeric part
                number_match = re.search(r'(-?\d[\d,]*\.?\d*)', model_answer)
                if number_match:
                    model_answer = number_match.group(1).replace(',', '')
            
            # If no unit words but has currency symbols, check if gold_answer also has currency symbols
            has_currency = bool(re.search(r'[$€£¥]', model_answer))
            
            # Check if gold_answer has currency symbols
            gold_has_currency = False
            if isinstance(gold_answer, str):
                gold_has_currency = bool(re.search(r'[$€£¥]', gold_answer))
            elif isinstance(gold_answer, list):
                for ans in gold_answer:
                    if isinstance(ans, str) and re.search(r'[$€£¥]', ans):
                        gold_has_currency = True
                        break
            
            # Decide whether to keep currency symbols based on gold_answer characteristics
            if has_currency and not gold_has_currency:
                model_answer = re.sub(r'[$€£¥]', '', model_answer)
            # Conversely, if gold_answer has currency symbols but model_answer doesn't, try using same currency symbol
            elif gold_has_currency and not has_currency:
                if isinstance(gold_answer, str):
                    currency_match = re.search(r'([$€£¥])', gold_answer)
                    if currency_match:
                        model_answer = currency_match.group(1) + model_answer
                elif isinstance(gold_answer, list) and len(gold_answer) > 0:
                    if isinstance(gold_answer[0], str):
                        currency_match = re.search(r'([$€£¥])', gold_answer[0])
                        if currency_match:
                            model_answer = currency_match.group(1) + model_answer
        
        # Ensure list format
        if not isinstance(model_answer, list):
            model_answer = [str(model_answer)]
    
    elif answer_type == "span":
        # Special handling for span type answers
        
        # Ensure gold_answer is in list format
        if not isinstance(gold_answer, list):
            gold_answer = [gold_answer]
        
        # Check if gold_answer contains currency symbols
        gold_has_currency = False
        gold_has_unit = False
        for ans in gold_answer:
            if isinstance(ans, str):
                if re.search(r'[$€£¥]', ans):
                    gold_has_currency = True
                if re.search(r'\b(thousand|million|billion|percent|%)\b', ans, re.IGNORECASE):
                    gold_has_unit = True
        
        # If model answer is string, perform special handling
        if isinstance(model_answer, str):
            has_currency = bool(re.search(r'[$€£¥]', model_answer))
            has_unit = bool(re.search(r'\b(thousand|million|billion|percent|%)\b', model_answer, re.IGNORECASE))
            
            # 1. If gold has symbols but prediction doesn't, try adding symbols
            if gold_has_currency and not has_currency:
                # Try to extract currency symbol from gold answer
                currency_symbol = ""
                for ans in gold_answer:
                    if isinstance(ans, str):
                        currency_match = re.search(r'([$€£¥])', ans)
                        if currency_match:
                            currency_symbol = currency_match.group(1)
                            break
                
                if currency_symbol:
                    # Add currency symbol
                    model_answer = currency_symbol + model_answer
            
            # 2. If gold has units but prediction doesn't, might need to add units
            if gold_has_unit and not has_unit:
                # Check if units need to be added
                for ans in gold_answer:
                    if isinstance(ans, str) and model_answer in ans:
                        # Exact matching substring, use complete gold answer
                        model_answer = ans
                        break
        
        # Ensure list format
        if not isinstance(model_answer, list):
            model_answer = [model_answer]
    
    # Default handling for other answer types
    else:
        if not isinstance(model_answer, list):
            model_answer = [model_answer]
        if not isinstance(gold_answer, list):
            gold_answer = [gold_answer]

    # Create normalized ground_truth dictionary
    eval_ground_truth = {
        "answer": gold_answer,
        "answer_type": answer_type,
        "scale": correct_scale
    }
    
    # Execute evaluation
    em_and_f1(
        ground_truth=eval_ground_truth,
        prediction=model_answer,
        pred_scale=correct_scale
    )
    
    # Get evaluation metrics
    exact_match, f1_score, scale_score, op_score = em_and_f1.get_overall_metric(reset=True)
    
    return model_answer, exact_match, f1_score


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


def compute_score(solution_str, table, paragraphs, ground_truth, extra_info=None, method='strict', 
                 format_score=0.1, score=1.0, return_details=False):
    """The scoring function for TaTQA task.
    
    Args:
        solution_str: the solution text from the model
        ground_truth: dictionary containing the correct answer, answer_type, and scale
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
        return_details: whether to return details of scoring (EM, F1, etc.)
    """
    # For debugging (print randomly to avoid overwhelming logs)
    do_print = random.randint(1, 64) == 1
    
    # Extract predicted answer
    model_answer = extract_answer(solution_str=solution_str)
    
    # First check if format is correct and get thinking score
    format_correct, thinking_score_val = check_format(solution_str)
    

    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {ground_truth}")
        print(f"Extracted answer: {model_answer}")
        print(f"Solution string: {solution_str}")
        print(f"Format correct: {format_correct}, Thinking score: {thinking_score_val}")

    if not format_correct:
        if do_print:
            print("Incorrect format: missing proper <think> or <answer> tags")
        
        if return_details:
            return {
                'combined_score': 0.0,
                'em': 0.0,
                'f1': 0.0,
                'format_correct': False
            }
        else:
            return 0.0
    
    # Check if answer content exists
    if model_answer is None:
        if return_details:
            return {
                'combined_score': thinking_score_val,
                'em': 0.0,
                'f1': 0.0,
                'format_correct': True
            }
        else:
            return thinking_score_val
    
    # Handle ground_truth
    if isinstance(ground_truth, dict) and "ground_truth" in ground_truth:
        ground_truth = ground_truth["ground_truth"]
    
    if not isinstance(ground_truth, dict):
        try:
            ground_truth = json.loads(ground_truth)
        except:
            if do_print:
                print(f"Unable to parse ground_truth: {ground_truth}")
            
            if return_details:
                return {
                    'combined_score': format_score + thinking_score_val,
                    'em': 0.0,
                    'f1': 0.0,
                    'format_correct': True
                }
            else:
                return format_score + thinking_score_val
    
    # Extract fields from ground_truth
    answer_type = ground_truth.get("answer_type", "")
    scale = ground_truth.get("scale", "")
    
    # Handle the answer field which could be a JSON string
    gold_answer = ground_truth.get("answer", "")
    if isinstance(gold_answer, str) and (gold_answer.startswith("[") or gold_answer.startswith("\"") or gold_answer.startswith("{")):
        try:
            gold_answer = json.loads(gold_answer)
        except:
            pass
    
    # Create complete ground truth dictionary for evaluation
    eval_ground_truth = {
        "answer": gold_answer,
        "answer_type": answer_type,
        "scale": scale
    }

    try:
        # Evaluate predicted answer
        model_answer, exact_match, f1_score = evaluate_prediction(
            model_answer,
            eval_ground_truth,
            scale=scale
        )
        
        if do_print:
            print(f"EM: {exact_match}, F1: {f1_score}")
            print(f"Processed prediction: {model_answer}")
            print(f"Correct answer: {gold_answer}")
        
        # If detailed information is requested
        if return_details:
            details = {
                'em': exact_match,
                'f1': f1_score,
                'format_correct': format_correct
            }
            
            # Calculate total score
            if exact_match > 0:
                # When answer is completely correct, give full score
                combined_score = score
            else:
                # When answer is incorrect, give base score + thinking score
                partial_score = format_score + thinking_score_val
                
                # If F1 score is high, give additional reward
                if f1_score > 0.5:
                    partial_score += 0.2
                    
                combined_score = min(0.6, partial_score)
            
            details['combined_score'] = combined_score
            return details
        
        # Otherwise, return regular score
        if exact_match > 0:
            # When answer is completely correct, give full score
            return score
        else:
            # When answer is incorrect
            partial_score = format_score + thinking_score_val
            
            # If F1 score is high, give additional reward
            if f1_score > 0.5:
                partial_score += 0.2
                
            return min(0.6, partial_score)
            
    except Exception as e:
        if do_print:
            print(f"Error during evaluation: {e}")
        
        if return_details:
            # When error occurs, only give base score
            combined_score = format_score + thinking_score_val
                
            return {
                'combined_score': combined_score,
                'em': 0.0,
                'f1': 0.0,
                'format_correct': format_correct,
                'error': str(e)
            }
        else:
            # When error occurs, only give base score
            combined_score = format_score + thinking_score_val
            return combined_score
