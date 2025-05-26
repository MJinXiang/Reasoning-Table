import re
import random
import json

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



def compute_score(predicted_answer, ground_truth):
    """The scoring function for WikiTableQuestions task.
    
    Args:
        predicted_answer: the predicted answer from the model
        ground_truth: string containing the correct answer
    """
    correct_answer = ground_truth

    # Check if answer matches ground truth
    normalized_predicted = normalize_answer(predicted_answer)
    normalized_ground_truth = normalize_answer(correct_answer)
    
    # Handle multiple possible answers separated by |
    if "|" in correct_answer:
        possible_answers = [normalize_answer(ans.strip()) for ans in correct_answer.split("|")]
        is_correct = normalized_predicted in possible_answers
    else:
        is_correct = normalized_predicted == normalized_ground_truth

    # Set accuracy - 1 if answer is correct, 0 otherwise
    accuracy = 1.0 if is_correct else 0.0

    return {
        'accuracy': accuracy,
    }