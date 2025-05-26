import re
import random
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import Dict, Any, List, Union
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

def extract_keywords(reference_text):
    """Extract keywords from reference answer"""
    if isinstance(reference_text, list):
        reference_text = reference_text[0] if reference_text else ""
    
    # Simple word tokenization
    words = re.findall(r'\b\w+\b', reference_text.lower())
    
    # Filter common stopwords
    stopwords = {"a", "an", "the", "in", "on", "at", "to", "for", "with", "by", "as", "and"}
    keywords = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Get proper nouns (capitalized words)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', reference_text)
    
    # Extract numbers
    numbers = re.findall(r'\b\d+(?:[-–—]\d+)?', reference_text)
    
    # Return all key information
    return list(set(keywords + proper_nouns + numbers))

def check_format(solution_str):
    """Check if the solution follows the required format with proper <think> and <answer> tags."""
    thinking = extract_thinking(solution_str)
    answer = extract_answer(solution_str)
    
    if thinking is None or answer is None:
        return False, 0
        
    if len(answer.strip()) < 1: 
        return False, 0
    
    # Check for reasoning steps
    steps = re.findall(r'(step \d+|first|second|then|next|finally)', thinking.lower())
    reasoning_score = min(0.1, len(steps) * 0.02)  # Max 0.1 points for reasoning
    
    return True, reasoning_score

def tokenize_sentence(sentence):
    """Tokenize a sentence into words and punctuation."""
    sentence = sentence.lower()
    tokens = re.findall(r'\w+|[^\w\s]', sentence)
    return tokens

def compute_bleu_score(prediction, reference):
    """Compute BLEU score between prediction and reference."""
    try:
        if isinstance(reference, list):
            if len(reference) == 0:
                return 0.0
            reference_text = reference[0]
        else:
            reference_text = reference
        
        prediction_tokens = tokenize_sentence(prediction)
        reference_tokens = [tokenize_sentence(reference_text)]
        
        smoothing = SmoothingFunction().method1
        weights = (0.4, 0.3, 0.2, 0.1)  # Focus more on 1-gram and 2-gram
        
        bleu_score = sentence_bleu(reference_tokens, prediction_tokens, 
                                  weights=weights, smoothing_function=smoothing)
        
        return bleu_score
    except Exception as e:
        print(f"Error computing BLEU score: {e}")
        return 0.0

def check_answer_quality(predicted_answer, ground_truth, question=""):
    """
    Check if the predicted answer meets the quality requirements for the FETAQA task.
    
    Returns a tuple of (is_valid, bleu_score)
    """
    # Check if it's a single sentence
    sentence_enders = re.findall(r'[.!?]', predicted_answer)
    if len(sentence_enders) > 1:
        sentences = re.split(r'[.!?]\s+', predicted_answer)
        non_empty_sentences = [s for s in sentences if s.strip()]
        if len(non_empty_sentences) > 1:
            return False, 0.0
    
    # Check if it contains lists
    if re.search(r'(\d+\.\s+|\-\s+|\*\s+)', predicted_answer):
        return False, 0.0
    
    # Check if it explicitly mentions highlighted cells
    highlight_mentions = ["highlight", "highlighted", "<hl>", "</hl>", "marked"]
    if any(mention in predicted_answer.lower() for mention in highlight_mentions):
        return False, 0.0
    
    # Question relevance check if question is provided
    penalty = 0.0
    if question:
        question_keywords = set(re.findall(r'\b\w+\b', question.lower()))
        question_keywords = {w for w in question_keywords if len(w) > 3 and w not in {"what", "when", "where", "which", "whom", "whose", "why", "how", "did", "does", "was", "were", "are", "can", "could", "should", "would"}}
        
        answer_text = predicted_answer.lower()
        
        contains_question_context = False
        for keyword in question_keywords:
            if keyword in answer_text or any(kw in answer_text for kw in [keyword + "s", keyword + "ed", keyword + "ing"]):
                contains_question_context = True
                break
        
        if not contains_question_context and len(question_keywords) > 0:
            penalty = 0.2
    
    bleu_score = compute_bleu_score(predicted_answer, ground_truth)
    
    return True, max(0, bleu_score - penalty)

def compute_score(solution_str, ground_truth, extra_info=None, format_score=0.1, min_score=0.0, max_score=1.0, return_details=False):
    """
    The scoring function for FETAQA task using BLEU.
    
    Args:
        solution_str: the solution text from the model
        ground_truth: string or dict containing the correct reference sentence(s)
        extra_info: additional information (not used in current implementation)
        format_score: base score for correct format
        min_score: minimum score to return
        max_score: maximum score to return
        return_details: whether to return detailed metrics
    """
    predicted_answer = extract_answer(solution_str=solution_str)
    
    # Handle ground_truth format
    if isinstance(ground_truth, dict) and "ground_truth" in ground_truth:
        correct_answer = ground_truth["ground_truth"]
    else:
        correct_answer = ground_truth
    
    # Handle possible multiple reference answers
    if isinstance(correct_answer, str) and correct_answer.startswith('[') and correct_answer.endswith(']'):
        try:
            import ast
            correct_answer = ast.literal_eval(correct_answer)
        except:
            pass

    # For debugging (print randomly to avoid overwhelming logs)
    do_print = random.randint(1, 32) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {correct_answer}")
        print(f"Extracted answer: {predicted_answer}")
        print(f"Solution string: {solution_str}")

    # First check if format is correct
    format_correct, reasoning_score = check_format(solution_str)

    if not format_correct:
        if do_print:
            print("Incorrect format: missing proper <think> or <answer> tags")
        if return_details:
            return {
                'combined_score': 0.0,
                'bleu': 0.0,
                'format_correct': False
            }
        return 0.0
    
    # Check if answer content is correct
    if predicted_answer is None:
        if do_print:
            print("No answer found in the proper format")
        if return_details:
            return {
                'combined_score': 0.0,
                'bleu': 0.0,
                'format_correct': True
            }
        return 0.0
    
    # Check answer quality and compute BLEU score
    is_valid, bleu_score = check_answer_quality(predicted_answer, correct_answer, "")
    
    if not is_valid:
        if do_print:
            print("Answer does not meet quality requirements")
        if return_details:
            return {
                'combined_score': 0.1,
                'bleu': 0.0,
                'format_correct': True
            }
        return 0.1
    
    # Set BLEU baseline and adjustment
    base_bleu = 0.05  # Minimum BLEU score
    adjusted_bleu = max(bleu_score, base_bleu)

    # Use higher answer score when BLEU score is greater than 0.5
    if bleu_score > 0.5:
        answer_score = bleu_score
        if do_print:
            print(f"High BLEU score detected: {bleu_score}")
    else:
        # Add keyword matching bonus
        keyword_bonus = 0.0
        keywords = extract_keywords(correct_answer)
        
        # Calculate keyword match rate
        matched_keywords = 0
        for keyword in keywords:
            if keyword.lower() in predicted_answer.lower():
                matched_keywords += 1
        
        if keywords:
            keyword_match_rate = matched_keywords / len(keywords)
            keyword_bonus = min(0.2, keyword_match_rate * 0.3)  # Max 0.2 points for keyword matching
    
        # Weight allocation
        answer_score = (reasoning_score * 0.05) + (adjusted_bleu * 0.75) + (keyword_bonus * 0.2)

    final_score = answer_score
    
    # Ensure score is within reasonable range
    final_score = min(max(final_score, min_score), max_score)
    
    if do_print:
        print(f"Answer score: {answer_score}")
        print(f"Final score: {final_score}")
    
    # Return detailed metrics if requested
    if return_details:
        return {
            'combined_score': final_score,
            'bleu': bleu_score,
            'format_correct': format_correct
        }
    
    return final_score
