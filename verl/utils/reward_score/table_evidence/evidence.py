import re

def extract_evidence_pairs(response_string):
    """
    Extract evidence pairs from response string.
    
    Args:
        response_string (str): Response string containing evidence pairs
    
    Returns:
        list: List of extracted evidence pairs, each pair is a dictionary containing value and column
    """
    # Match <|value|><|column|>. or <|value|><|column|>.: format
    pattern = r'<\|(.*?)\|><\|(.*?)\|>'
    matches = re.findall(pattern, response_string)
    unique_pairs = set()
    evidence_pairs = []
    for value, column in matches:
        if (value, column) in unique_pairs:
            continue
        unique_pairs.add((value, column))
        evidence_pairs.append({
            "value": value,
            "column": column
        })
    
    return evidence_pairs
    
def compute_evidence_intersection_union(evidence_pairs1, evidence_pairs2):
    """
    Compute the intersection and union size of two sets of evidence pairs.
    
    Args:
        evidence_pairs1 (list): First set of evidence pairs
        evidence_pairs2 (list): Second set of evidence pairs
    
    Returns:
        tuple: (intersection size, union size)
    """
    # Convert evidence pairs to hashable tuple sets
    set1 = {(pair['value'], pair['column']) for pair in evidence_pairs1}
    set2 = {(pair['value'], pair['column']) for pair in evidence_pairs2}
    
    # Compute intersection
    intersection = set1.intersection(set2)
    
    # Compute union
    union = set1.union(set2)
    
    return len(intersection), len(union)

# Optional: compute Jaccard similarity
def compute_jaccard_similarity(evidence_pairs1, evidence_pairs2):
    """
    Compute Jaccard similarity of two sets of evidence pairs (intersection size / union size).
    
    Args:
        evidence_pairs1 (list): First set of evidence pairs
        evidence_pairs2 (list): Second set of evidence pairs
    
    Returns:
        float: Jaccard similarity, range [0,1]
    """
    intersection_size, union_size = compute_evidence_intersection_union(
        evidence_pairs1, evidence_pairs2
    )
    
    # Avoid division by zero
    if union_size == 0:
        return 0.0
    
    return intersection_size / union_size

if __name__ == "__main__":
    test_string = '''Some text <|8,189|><|yds|>.: More text <|8,189|><|yds|>.'''
    
    result = extract_evidence_pairs(test_string)
    print(result)
    # Output: [{'value': '8,189', 'column': 'yds'}, {'value': '8,189', 'column': 'yds'}]

    evidence1 = [
            {"value": "8,189", "column": "yds"},
            {"value": "50", "column": "TD"},
            {"value": "42", "column": "games"}
        ]
    
    evidence2 = [
        {"value": "8,189", "column": "yds"},
        {"value": "42", "column": "games"},
        {"value": "11", "column": "int"}
    ]
    
    intersection_size, union_size = compute_evidence_intersection_union(evidence1, evidence2)
    print(f"Intersection size: {intersection_size}")  # Should be 2
    print(f"Union size: {union_size}")        # Should be 4
    
    similarity = compute_jaccard_similarity(evidence1, evidence2)
    print(f"Jaccard similarity: {similarity:.2f}")  # Should be 0.50