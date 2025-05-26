import re
import random
from typing import List, Dict, Any, Tuple
from collections import Counter
import json
import os
import datetime
import concurrent.futures
import logging
from llm import call_api_with_retry, initialize_client
from prompt import LLM_EVAL


# Set up general logging
logger = logging.getLogger('hitab_eval')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Set up LLM judgment dedicated logging
llm_judge_logger = logging.getLogger('hitab_llm_judge')
if not llm_judge_logger.handlers:
    llm_judge_logger.setLevel(logging.INFO)
    # Create log directory
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    # Create log file, using current date as filename
    today = datetime.datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(log_dir, f'hitab_llm_judge_{today}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    llm_judge_logger.addHandler(file_handler)
    # Also output to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(file_formatter)
    llm_judge_logger.addHandler(console)

# Global LLM client, initialize only once
GLOBAL_LLM_CLIENT = initialize_client({
    "model_type": "claude",
    "model_path": "claude-3-7-sonnet-2025021"
})

# Global thread pool for parallel processing
GLOBAL_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=10)

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

def check_format(solution_str):
    """Check if the solution follows the required format with proper <think> and <answer> tags."""
    thinking = extract_thinking(solution_str)
    answer = extract_answer(solution_str)
    
    if thinking is None or answer is None:
        return False, 0
        
    if len(answer.strip()) < 1: 
        return False, 0
    
    thinking_score = 0 
    
    return True, thinking_score

def check_answer_correctness(model_answer, expected_answer):
    """
    Check if the model's answer is correct compared to the expected answer.
    Handles various formats including lists, single values, and numerical comparisons.
    """
    import re  # Ensure module is imported
    
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

def evaluate_with_llm(question, candidate_answer, correct_answer, client_info=None):
    """Use LLM to evaluate if answers are consistent"""
    try:
        # If no client provided, use global client
        if client_info is None:
            client_info = GLOBAL_LLM_CLIENT
            
        # Construct prompt using template
        prompt = LLM_EVAL.format(
            question=question,
            candidate_answer=candidate_answer,
            correct_answer=correct_answer
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant specializing in answer evaluation."},
            {"role": "user", "content": prompt}
        ]
        
        # Call LLM API
        success, response = call_api_with_retry(
            client_info=client_info, 
            messages=messages,
            max_tokens=100,  # Only need short replies
            temperature=0.1  # Low temperature for consistent replies
        )
        
        if not success:
            logger.warning(f"LLM API call failed: {response}")
            return False  # API call failed, default to inconsistent
        
        # Handle different API return formats
        if isinstance(response, str):
            answer_text = response
        else:
            # Assume OpenAI format response
            try:
                answer_text = response.choices[0].message.content
            except:
                logger.warning(f"Cannot parse LLM response: {response}")
                return False
        
        # Check if answer contains "Yes"
        is_consistent = "yes" in answer_text.lower()
        
        # Log to dedicated LLM judgment log
        log_message = (f"QUESTION: {question}\n"
                       f"CANDIDATE: {candidate_answer}\n"
                       f"CORRECT: {correct_answer}\n"
                       f"LLM JUDGMENT: {'CONSISTENT' if is_consistent else 'INCONSISTENT'}\n"
                       f"RAW RESPONSE: {answer_text}\n"
                       f"{'='*50}")
        llm_judge_logger.info(log_message)
        
        return is_consistent
        
    except Exception as e:
        logger.error(f"Error during LLM evaluation: {e}")
        return False  # Default to inconsistent when error occurs

def compute_score(solution_str, ground_truth, extra_info=None, format_score=0.1, score=1.0, 
                 return_details=False, llm_client_info=None, max_workers=100):
    """
    The scoring function for HiTAB task, with parallel LLM evaluation.
    Can handle single scoring request or batch scoring requests.
    
    Args:
        solution_str: Single solution string or list of solution strings
        ground_truth: Standard answer or list of standard answers
        extra_info: Additional information or list of additional information
        format_score: Base score for correct format
        score: Full score for correct answer
        return_details: Whether to return detailed scoring information
        llm_client_info: LLM client information
        max_workers: Maximum number of worker threads for parallel processing
        
    Returns:
        Single score or list of scores
    """
    # Detect if this is a batch processing request
    is_batch = isinstance(solution_str, list)
    
    # If it's a batch processing request, use thread pool for parallel processing
    if is_batch:
        # Prepare batch processing parameters
        solutions = solution_str
        
        # Ensure other parameters adapt to batch processing
        if not isinstance(ground_truth, list):
            ground_truths = [ground_truth] * len(solutions)
        else:
            ground_truths = ground_truth
            
        if not isinstance(extra_info, list):
            extra_infos = [extra_info] * len(solutions)
        else:
            extra_infos = extra_info
        
        # Use thread pool for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, solution in enumerate(solutions):
                gt = ground_truths[i] if i < len(ground_truths) else None
                info = extra_infos[i] if i < len(extra_infos) else None
                
                futures.append(
                    executor.submit(
                        compute_score,  # Recursively call itself, but pass single instance
                        solution,
                        gt,
                        info,
                        format_score, score, return_details, llm_client_info
                    )
                )
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Score computation failed: {e}")
                    results.append(0.0 if not return_details else {"error": str(e)})
            
            return results
    
    # Single instance scoring processing logic
    predicted_answer = extract_answer(solution_str=solution_str)
    
    # Get question text (if exists)
    question = extra_info.get("question", "") if extra_info and isinstance(extra_info, dict) else ""
    
    # Handle ground_truth format
    if isinstance(ground_truth, dict) and "ground_truth" in ground_truth:
        correct_answer = ground_truth["ground_truth"]
    else:
        correct_answer = ground_truth

    # For debugging (print randomly to avoid overwhelming logs)
    do_print = random.randint(1, 32) == 1
    
    if do_print:
        logger.info(f"--------------------------------")
        logger.info(f"Ground truth: {correct_answer}")
        logger.info(f"Extracted answer: {predicted_answer}")
        logger.info(f"Solution string: {solution_str}")

    # First check if format is correct and get thinking score
    format_correct, thinking_score = check_format(solution_str)

    # Initialize accuracy metric
    accuracy = 0.0  # Default to 0 indicating incorrect

    if not format_correct:
        if do_print:
            logger.info("Incorrect format: missing proper <think> or <answer> tags")
            
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
            logger.info("No answer found in the proper format")
            
        if return_details:
            return {
                'combined_score': thinking_score,
                'accuracy': accuracy,
                'format_correct': format_correct
            }
        return thinking_score 
    
    # Check if the answer is correct - exact answer evaluation only
    is_correct = check_answer_correctness(predicted_answer, correct_answer)
    
    # Set accuracy
    accuracy = 1.0 if is_correct else 0.0
    
    # If answer is judged incorrect and there's question text, perform secondary evaluation (using parallel processing)
    llm_verified = False
    if not is_correct and question:
        # Use parallel LLM evaluation - use global thread pool, support true parallelism
        future = GLOBAL_THREAD_POOL.submit(
            evaluate_with_llm,
            question=question, 
            candidate_answer=predicted_answer,
            correct_answer=correct_answer,
            client_info=llm_client_info or GLOBAL_LLM_CLIENT  # Use passed client or global client
        )
            
        try:
            # Get evaluation result
            llm_consistent = future.result(timeout=30)  # Add timeout control
            
            if llm_consistent:
                is_correct = True
                accuracy = 1.0
                llm_verified = True
                if do_print:
                    logger.info(f"LLM verified the answer as correct!")
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            # Continue using original scoring result
            llm_verified = False

    # Calculate final score (removed evidence scoring)
    final_score = 0.0
    if is_correct:
        if do_print:
            logger.info("Correct answer" + (" (LLM verified)" if llm_verified else ""))
        final_score = score
    else:
        if do_print:
            logger.info(f"Correct format but incorrect answer, thinking score: {thinking_score}")
        final_score = format_score + thinking_score
    
    # If detailed metrics are needed
    if return_details:
        details = {
            'combined_score': final_score,
            'accuracy': accuracy,
            'format_correct': format_correct
        }
        if llm_verified:
            details['llm_verified'] = True
        return details
    
    return final_score
