import random
import numpy as np
from query_calculate import exact_count, sample_count, exact_sum, sample_sum

# Paper mentioned they bound the query so that the exact result will not be zero most of the time 
def generate_bounded_random_query(data, dimensions, test = False):
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
        
        # Ensure lower < upper
        if lower >= upper:
            lower, upper = right_min, right_max  # fallback
        
        predicates[dim] = (lower, upper)
    
    return predicates

# Random query for demo, totally random
def generate_random_query(data, dimensions, test=False):
    predicates = {}
    for dim in dimensions:
        min_val = data[dim].min()
        max_val = data[dim].max()

        a = random.uniform(min_val, max_val)
        b = random.uniform(min_val, max_val)

        lower, upper = min(a, b), max(a, b)

        if test:
            print(f"dim: {dim}, lower: {lower}, upper: {upper}")

        predicates[dim] = (lower, upper)

    return predicates

def generate_query_log(
    num_queries,
    data,
    sample,
    dimensions,
    full_data_size,
    agg_type,               # aggregation type (SUM or COUNT)
    agg_col=None,           # when agg_type is SUM then give agg_col (power dataset's agg_col is Global_active_power)
    min_exact=1,            # if None means need to calculate avg exact
    min_estimate=0,         # default 0 to ensure there is valuable result
    max_attempts=10000,     # setup max attempts
    compute_avg_exact=False # whether to calculate avg_exact (usually for the first time)
):
    print(f"Generating {agg_type} query log...")
    
    if agg_type == 'SUM' and agg_col is None:
        raise ValueError("agg_col must be provided when agg_type is SUM")
    
    query_log = []
    attempts = 0
    exact_values = []

    while len(query_log) < num_queries and attempts < max_attempts:
        attempts += 1
        q = generate_bounded_random_query(data, dimensions)
        
        if agg_type == 'COUNT':
            exact = exact_count(q, data)
            estimate = sample_count(q, sample, full_data_size)
        else:  # SUM
            exact = exact_sum(agg_col, q, data)
            estimate = sample_sum(agg_col, q, sample, full_data_size)
        
        if exact > min_exact and estimate > min_estimate:
            error = exact - estimate
            query_log.append({
                'query': q,
                'exact': exact,
                'estimate': estimate,
                'error': error
            })
            exact_values.append(exact)
    
    print(f"Generated {len(query_log)} queries after {attempts} attempts")
    
    if len(query_log) < num_queries:
        print(f"Warning: Only generated {len(query_log)}/{num_queries} queries within {max_attempts} attempts")
    
    if compute_avg_exact:
        avg_exact = np.mean(exact_values) if exact_values else 0
        print(f"Average exact {agg_type}: {avg_exact:.2f}\n")
        return avg_exact, query_log
    else:
        return query_log
