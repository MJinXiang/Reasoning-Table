import re
import random
import json

def extract_sql(solution_str):
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


def parse_sql(query, columns):
    """Parse SQL query into structured components."""
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']

    columns = list(columns)
    
    # Create column mapping for case-insensitive matching
    col_map = {}
    for i, col in enumerate(columns):
        norm_col = col.lower().strip().replace('(', '').replace(')', '').replace(' ', '')
        col_map[norm_col] = i
    
    # Extract SELECT clause
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
    select_clause = select_match.group(1).strip() if select_match else None
    
    if select_clause is None:
        print(f"SELECT clause not found in query: {query}")
        return {
            "agg": 0,
            "conds": {"column_index": [], "condition": [], "operator_index": []},
            "human_readable": query,
            "sel": -1
        }
    
    # Parse aggregation function
    agg = 0
    select_col = select_clause
    for i, op in enumerate(agg_ops[1:], 1):
        pattern = rf'{op}\s*\((.*?)\)'
        match = re.search(pattern, select_clause, re.IGNORECASE)
        if match:
            agg = i
            select_col = match.group(1).strip()
            break
    
    # Parse SELECT column
    sel = -1
    if select_col in columns:
        sel = columns.index(select_col)
    else:
        select_col_norm = select_col.lower().strip().replace('(', '').replace(')', '').replace(' ', '')
        if select_col_norm in col_map:
            sel = col_map[select_col_norm]

    # Parse WHERE conditions
    conds = {"column_index": [], "condition": [], "operator_index": []}
    where_match = re.search(r'WHERE\s+(.*)', query, re.IGNORECASE)
    
    if where_match:
        conditions_str = where_match.group(1).strip()
        conditions = re.split(r'\s+AND\s+|\s+OR\s+', conditions_str, flags=re.IGNORECASE)
        
        for cond in conditions:
            for i, op in enumerate(cond_ops):
                if op in cond:
                    parts = cond.split(op, 1)
                    if len(parts) == 2:
                        col_name = parts[0].strip()
                        value = parts[1].strip().strip("'").strip('"')
                        
                        if col_name in columns:
                            conds["column_index"].append(columns.index(col_name))
                            conds["condition"].append(value)
                            conds["operator_index"].append(i)
                        else:
                            col_norm = col_name.lower().replace(' ', '').replace('(', '').replace(')', '')
                            if col_norm in col_map:
                                conds["column_index"].append(col_map[col_norm])
                                conds["condition"].append(value)
                                conds["operator_index"].append(i)
                    break
    
    return {
        "agg": agg,
        "conds": conds,
        "human_readable": query,
        "sel": sel
    }

def normalize_sql(sql_string):
    """Normalize SQL query by removing extra spaces and converting to lowercase."""
    sql_string = ' '.join(sql_string.split()).lower()
    sql_string = sql_string.replace("'", "").replace('"', "")  

    agg_funcs = ['count', 'sum', 'avg', 'min', 'max']
    
    for func in agg_funcs:
        pattern = r'{}[ ]*\(([^)]+)\)'.format(func)
        replacement = r'{} \1'.format(func)
        sql_string = re.sub(pattern, replacement, sql_string)

    return sql_string

def score_sql(parsed_sql, correct_sql):
    """Calculate partial score for SQL components."""
    weights = {
        "sel": 0.2,  
        "agg": 0.2, 
        "conds_column": 0.2,  
        "conds_value": 0.2, 
        "conds_operator": 0.2, 
    }

    score = 0

    # Check SELECT column
    if parsed_sql["sel"] == correct_sql["sel"]:
        score += weights["sel"] * 1

    # Check aggregation function
    if parsed_sql["agg"] == correct_sql["agg"]:
        score += weights["agg"] * 1

    parsed_conds = parsed_sql["conds"]
    correct_conds = correct_sql["conds"]

    def unordered_match(parsed_list, correct_list):
        return sorted(parsed_list) == sorted(correct_list)

    # Check condition columns
    if unordered_match(parsed_conds["column_index"], correct_conds["column_index"]):
        score += weights["conds_column"] * 1

    # Check condition values
    if unordered_match(parsed_conds["condition"], correct_conds["condition"]):
        score += weights["conds_value"] * 1

    # Check condition operators
    if unordered_match(parsed_conds["operator_index"], correct_conds["operator_index"]):
        score += weights["conds_operator"] * 1

    return round(score, 2)

def compute_exact_accuracy(parsed_sql, correct_sql):
    """
    Calculate accuracy: returns 1 only if all components match, otherwise returns 0.
    No weights used, all components must match to be considered correct.
    
    Args:
        parsed_sql: Parsed predicted SQL
        correct_sql: Parsed correct SQL
    
    Returns:
        1: if all components match
        0: if any component doesn't match
    """
    # Check if SELECT column matches
    if parsed_sql["sel"] != correct_sql["sel"]:
        return 0
    
    # Check if aggregation function matches
    if parsed_sql["agg"] != correct_sql["agg"]:
        return 0
    
    parsed_conds = parsed_sql["conds"]
    correct_conds = correct_sql["conds"]
    
    def unordered_match(parsed_list, correct_list):
        return sorted(parsed_list) == sorted(correct_list)
    
    # Check if condition columns match
    if not unordered_match(parsed_conds["column_index"], correct_conds["column_index"]):
        return 0
    
    # Check if condition values match
    if not unordered_match(parsed_conds["condition"], correct_conds["condition"]):
        return 0
    
    # Check if condition operators match
    if not unordered_match(parsed_conds["operator_index"], correct_conds["operator_index"]):
        return 0
    
    # All components match, return 1
    return 1


def compute_score(solution_str, ground_truth, table, ans, format_score=0.1, score=1.0, return_details=False):
    """
    The scoring function for WikiSQL.

    Args:
        solution_str: The predicted SQL query string.
        ground_truth: A dictionary containing the ground truth SQL information, including the human-readable SQL query.
        table: The database table information.
        ans: The correct answer structure.
        format_score: The score for correct format but wrong answer.
        score: The score for the correct answer.
        return_details: Whether to return detailed metrics including accuracy
    """
    ground_truth_sql = ground_truth
    
    # Detect and parse JSON strings
    if isinstance(ans, str):
        try:
            if ans.startswith('{') or ans.startswith('['):
                ground_truth_answer = json.loads(ans)
            else:
                ground_truth_answer = ans
        except:
            print(f"Unable to parse ans as JSON: {ans}")
            ground_truth_answer = ans  # Use original string if parsing fails
    else:
        ground_truth_answer = ans

    if isinstance(table, str):
        try:
            if table.startswith('{') or table.startswith('['):
                table = json.loads(table)
        except:
            print(f"Unable to parse table as JSON: {table}")
    else:
        table = table

    # Initialize accuracy metric
    accuracy = 0.0  # Default to 0 indicating incorrect
    
    predicted_sql = extract_sql(solution_str=solution_str)
    
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Ground Truth SQL: {ground_truth_sql}")
        print(f"Generated SQL: {predicted_sql}")
        print(f"Solution string: {solution_str}")
        
    # If unable to extract SQL statement, return 0 score
    if predicted_sql is None:
        if do_print:
            print(f"No SQL query found")
            
        if return_details:
            return {
                'combined_score': 0.0,
                'accuracy': accuracy,
                'format_correct': False
            }
        return 0
    else:
        # Parse the predicted SQL
        predicted_answer = parse_sql(predicted_sql, table['header'])

        # Compare after SQL normalization
        predicted_sql_normalize = normalize_sql(predicted_sql)
        ground_truth_sql_normalize = normalize_sql(ground_truth_sql)
        
        # First check if normalized SQL strings are directly equal
        if predicted_sql_normalize == ground_truth_sql_normalize:
            if do_print:
                print(f"Correct SQL query: PredSQL: {predicted_sql}, GoldSQL: {ground_truth_sql}")
                
            # Exact match, accuracy is 1
            accuracy = 1.0
            final_score = score
            
            if return_details:
                return {
                    'combined_score': final_score,
                    'accuracy': accuracy,
                    'format_correct': True
                }
            return final_score
        else:
            # SQL strings not directly equal, need structural comparison
            # Calculate partial score (to reward predictions close to correct answer)
            partial_score = score_sql(predicted_answer, ground_truth_answer)
            
            # Use new accuracy calculation method, only correct if all components match
            accuracy = compute_exact_accuracy(predicted_answer, ground_truth_answer)
            
            if do_print:
                print(f"Partial score: {partial_score}, Accuracy: {accuracy}")
                print(f"Predicted Answer: {predicted_answer}")
                print(f"Truth Answer: {ground_truth_answer}")
                print(f"Table headers: {table['header']}")
                
            if return_details:
                return {
                    'combined_score': partial_score,
                    'accuracy': accuracy,
                    'format_correct': True
                }
            return partial_score
