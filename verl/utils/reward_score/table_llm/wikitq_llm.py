import re
import random
import json
import os
import datetime
import concurrent.futures
import logging
from llm import call_api_with_retry, initialize_client
from prompt import LLM_EVAL


# Set up general logging
logger = logging.getLogger('wikitq_eval')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Set up LLM judgment dedicated logging
llm_judge_logger = logging.getLogger('wikita_llm_judge')
if not llm_judge_logger.handlers:
    llm_judge_logger.setLevel(logging.INFO)
    # Create log directory
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    # Create log file, using current date as filename
    today = datetime.datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(log_dir, f'wikitq_llm_judge_{today}.log')
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
    """Normalize the answer string for comparison."""
    if answer is None:
        return None
    
    answer = answer.lower()
    
    answer = re.sub(r'[^\w\s]', ' ', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    if answer.isdigit():
        answer = str(int(answer))
    
    return answer

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

def compute_score(solution_str, ground_truth, extra_info=None, method='strict', 
                 format_score=0.1, score=1.0, return_details=False, llm_client_info=None,
                 max_workers=100):
    """
    The scoring function for WikiTableQuestions task, with parallel LLM evaluation.
    Can handle single scoring request or batch scoring requests.
    
    Args:
        solution_str: Single solution string or list of solution strings
        ground_truth: Single standard answer or list of standard answers
        extra_info: Single additional information or list of additional information
        method: Scoring method
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
        
        # Ensure ground_truth and extra_info adapt to batch processing
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
                        method, format_score, score, return_details, llm_client_info
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
    
    # Following is the single instance processing logic (original code)
    # Get question text (if exists)
    question = extra_info.get("question", "") if extra_info and isinstance(extra_info, dict) else ""
    
    predicted_answer = extract_answer(solution_str=solution_str)
    correct_answer = ground_truth
    
    # For debugging (print randomly to avoid overwhelming logs)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        logger.info(f"--------------------------------")
        logger.info(f"Ground truth: {correct_answer}")
        logger.info(f"Extracted answer: {predicted_answer}")
        logger.info(f"Solution string: {solution_str}")

    # Initialize accuracy metric
    accuracy = 0.0  # Default to 0 indicating incorrect
    
    # First check if format is correct and get thinking score
    format_correct, thinking_score = check_format(solution_str)

    if not format_correct:
        if do_print:
            logger.info("Incorrect format: missing proper <think> or <answer> tags")
        
        if return_details:
            return {
                'combined_score': 0.0,
                'accuracy': accuracy,
                'format_correct': False
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
                'format_correct': True
            }
        return thinking_score 
    
    # Check if answer matches ground truth
    normalized_predicted = normalize_answer(predicted_answer)
    normalized_ground_truth = normalize_answer(correct_answer)
    
    # Handle multiple possible answers separated by |
    if "|" in correct_answer:
        possible_answers = [normalize_answer(ans.strip()) for ans in correct_answer.split("|")]
        is_correct = normalized_predicted in possible_answers
    else:
        is_correct = normalized_predicted == normalized_ground_truth

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

    # Calculate final score - removed evidence scoring related logic
    final_score = 0.0
    if is_correct:
        if do_print:
            logger.info("Correct answer" + (" (LLM verified)" if llm_verified else ""))
        final_score = score
    else:
        if do_print:
            logger.info(f"Incorrect answer, thinking score: {thinking_score}")
        final_score = format_score + thinking_score
    
    # If detailed metrics are needed
    if return_details:
        details = {
            'combined_score': final_score,
            'accuracy': accuracy,
            'format_correct': True
        }
        if llm_verified:
            details['llm_verified'] = True
        return details
    
    return final_score
