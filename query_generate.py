import random
from query_calculate import exact_count, sample_count

def generate_random_query(data, dimensions, test = False):
    """
    Generate query ranges exactly as described in the LAQP paper for POWER dataset:
    - Left boundary: uniform from [min, min + (max-min)/4]  → first quarter
    - Right boundary: uniform from [max - (max-min)/4, max] → last quarter
    This ensures lower < upper with high probability and avoids empty results.
    """
    predicates = {}
    for dim in dimensions:
        min_val = data[dim].min()
        max_val = data[dim].max()
        if test : print(f"dim: {dim}, min value: {min_val}, max value: {max_val}")
        range_width = max_val - min_val
        
        # First quarter for left boundary
        left_min = min_val
        left_max = min_val + range_width / 4
        lower = random.uniform(left_min, left_max)
        
        # Last quarter for right boundary
        right_min = max_val - range_width / 4
        right_max = max_val
        upper = random.uniform(right_min, right_max)
        
        # Ensure lower < upper (very rare failure, but safe)
        if lower >= upper:
            lower, upper = right_min, right_max  # fallback
        
        predicates[dim] = (lower, upper)
    
    return predicates

def generate_query_log(num_queries, data, sample, dimensions, full_data_size):
    # Generate Query Log (e.g., 2000 for diversification)
    query_log = []
    while len(query_log) < num_queries:
        q = generate_random_query(data, dimensions)
        exact = exact_count(q, data)
        estimate = sample_count(q, sample, full_data_size)
        if exact > 100 and estimate > 0:  # Skip zeros
            error = exact - estimate
            query_log.append({'query': q, 'exact': exact, 'estimate': estimate, 'error': error})
    print(f"Generated {len(query_log)} queries")
    return query_log