import re
import random
import json
import difflib
from typing import Dict, List, Union, Any

from ..tatqa_utils.tatqa_metric import TaTQAEmAndF1, normalize_multi_span
from ..tatqa_utils.tatqa_utils import normalize_answer, is_number, to_number, scale_to_num



def compute_text_similarity(text1, text2):
    """
    Compute similarity between two texts
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score (0-1)
    """
    if not text1 or not text2:
        return 0
    
    # Convert text to lowercase and remove extra spaces
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    # Use difflib's SequenceMatcher to compute similarity
    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    return similarity

    
def evaluate_prediction(predicted_answer, ground_truth_dict, scale=None):
    """Evaluate the model answer against the ground truth."""
    # Initialize evaluator
    em_and_f1 = TaTQAEmAndF1()
    
    # Extract fields
    answer_type = ground_truth_dict.get("answer_type", "")
    gold_answer = ground_truth_dict.get("answer", "")
    correct_scale = scale if scale is not None else ground_truth_dict.get("scale", "")
    
    # Store original answer before preprocessing for potential debugging
    original_model_answer = predicted_answer
    
    # Handle possible JSON strings in ground_truth
    if isinstance(gold_answer, str) and (gold_answer.startswith("[") or gold_answer.startswith("\"") or gold_answer.startswith("{")):
        try:
            gold_answer = json.loads(gold_answer)
        except:
            pass

    # Preprocess model answer based on answer type and standard answer format
    if answer_type == "multi-span" and not isinstance(predicted_answer, list):
        predicted_answer = normalize_multi_span(predicted_answer)
    
    elif answer_type == "count":
        # Special handling for count type
        if isinstance(predicted_answer, list) and len(predicted_answer) > 0:
            predicted_answer = predicted_answer[0]
        
        # Clean symbols and convert to integer
        try:
            # Remove currency symbols and thousand separators
            clean_answer = re.sub(r'[$€£¥]', '', str(predicted_answer))
            clean_answer = clean_answer.replace(',', '')
            predicted_answer = str(int(float(clean_answer)))
        except:
            predicted_answer = str(predicted_answer)
                
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
        if isinstance(predicted_answer, str):
            # Check if it contains unit words
            has_unit = bool(re.search(r'\b(thousand|million|billion|percent|%)\b', predicted_answer, re.IGNORECASE))
            
            # If answer contains unit words and scale also specifies units, extract numeric part
            if has_unit and correct_scale:
                # Extract numeric part
                number_match = re.search(r'(-?\d[\d,]*\.?\d*)', predicted_answer)
                if number_match:
                    predicted_answer = number_match.group(1).replace(',', '')
            
            # If no unit words but has currency symbols, check if gold_answer also has currency symbols
            has_currency = bool(re.search(r'[$€£¥]', predicted_answer))
            
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
                predicted_answer = re.sub(r'[$€£¥]', '', predicted_answer)
            # Conversely, if gold_answer has currency symbols but predicted_answer doesn't, try using same currency symbol
            elif gold_has_currency and not has_currency:
                if isinstance(gold_answer, str):
                    currency_match = re.search(r'([$€£¥])', gold_answer)
                    if currency_match:
                        predicted_answer = currency_match.group(1) + predicted_answer
                elif isinstance(gold_answer, list) and len(gold_answer) > 0:
                    if isinstance(gold_answer[0], str):
                        currency_match = re.search(r'([$€£¥])', gold_answer[0])
                        if currency_match:
                            predicted_answer = currency_match.group(1) + predicted_answer
        
        # Ensure list format
        if not isinstance(predicted_answer, list):
            predicted_answer = [str(predicted_answer)]
    
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
        if isinstance(predicted_answer, str):
            has_currency = bool(re.search(r'[$€£¥]', predicted_answer))
            has_unit = bool(re.search(r'\b(thousand|million|billion|percent|%)\b', predicted_answer, re.IGNORECASE))
            
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
                    predicted_answer = currency_symbol + predicted_answer
            
            # 2. If gold has units but prediction doesn't, might need to add units
            if gold_has_unit and not has_unit:
                # Check if units need to be added
                for ans in gold_answer:
                    if isinstance(ans, str) and predicted_answer in ans:
                        # Exact matching substring, use complete gold answer
                        predicted_answer = ans
                        break
        
        # Ensure list format
        if not isinstance(predicted_answer, list):
            predicted_answer = [predicted_answer]
    
    # Default handling for other answer types
    else:
        if not isinstance(predicted_answer, list):
            predicted_answer = [predicted_answer]
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
        prediction=predicted_answer,
        pred_scale=correct_scale
    )
    
    # Get evaluation metrics
    exact_match, f1_score, scale_score, op_score = em_and_f1.get_overall_metric(reset=True)
    
    return predicted_answer, exact_match, f1_score


def compute_score(predicted_answer, ground_truth):
    """The scoring function for TaTQA task.
    
    Args:
        predicted_answer: the predicted answer from the model
        ground_truth: dictionary containing the correct answer, answer_type, and scale
        return_details: whether to return details of scoring (EM, F1, etc.)
    """
    
    # Handle ground_truth
    if isinstance(ground_truth, dict) and "ground_truth" in ground_truth:
        ground_truth = ground_truth["ground_truth"]
    else:
        try:
            ground_truth = json.loads(ground_truth)
        except:
            raise ValueError("Invalid ground_truth format")
    
    
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

    predicted_answer, exact_match, f1_score = evaluate_prediction(
        predicted_answer,
        eval_ground_truth,
        scale=scale
    )
    return {
            'em': exact_match,
            'f1': f1_score,
        }