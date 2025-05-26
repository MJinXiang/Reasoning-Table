import random
from .utils import extract_answer,check_format
from .evidence import extract_evidence_pairs, compute_jaccard_similarity
import json

def compute_score(data_source, solution_str, ground_truth, evidence, return_details=False):
    predicted_answer = extract_answer(solution_str=solution_str)
    format_score,_ = check_format(solution_str)
    if predicted_answer:
        if data_source == "hitab":
            from .hitab import compute_score as hitab_compute_score
            result = hitab_compute_score(predicted_answer, ground_truth)
            answer_score = result["accuracy"]
        elif data_source == "wikitq":
            from .wikitq import compute_score as wikitq_compute_score
            result = wikitq_compute_score(predicted_answer, ground_truth)
            answer_score = result["accuracy"]
        elif data_source == "tatqa":
            from .tatqa import compute_score as tatqa_compute_score
            result = tatqa_compute_score(predicted_answer, ground_truth)
            answer_score = result["em"] if result["em"] else result["f1"] if result["f1"] > 0.5 else 0
    else:
        result={}
        answer_score = 0
    
    predicted_evidence = extract_evidence_pairs(solution_str)

    if isinstance(evidence, str):
        evidence = json.loads(evidence)


    evidence_score = compute_jaccard_similarity(predicted_evidence, evidence)

    combined_score = answer_score*(1+0.5*evidence_score)+0.1*format_score

    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {ground_truth}")
        print(f"Extracted answer: {predicted_answer}")
        print(f"Solution string: {solution_str}")
        print(f"Answer score: {answer_score}")
        print(f"Evidence score: {evidence_score}")
        print(f"Format score: {format_score}")
        print(f"Combined score: {combined_score}")
    
    if return_details:
        result["answer_score"] = answer_score
        result["evidence_score"] = evidence_score
        result["format_score"] = format_score
        result["combined_score"] = combined_score
        return result

    return combined_score