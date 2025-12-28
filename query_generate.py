import random
import numpy as np
from query_calculate import exact_count, sample_count, exact_sum, sample_sum

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

def generate_uber_query_log(num_queries, data, sample, dimensions, full_data_size):
    attempts = 0
    max_attempts = 10000

    print("Generating uber query log...")
    query_log = []
    while len(query_log) < num_queries and attempts < max_attempts:
        q = generate_bounded_random_query(data, dimensions)
        exact = exact_count(q, data)
        estimate = sample_count(q, sample, full_data_size)
        if exact > 100 and estimate > 0:  # make sure valuable
            error = exact - estimate
            query_log.append({'query': q, 'exact': exact, 'estimate': estimate, 'error': error})
        attempts+=1
    print(f"Generated {len(query_log)} queries in {attempts} attempts")
    return query_log

def generate_power_query_log(agg_col, num_queries, data, sample, dimensions, full_data_size, avg_exact=0):
    attempts = 0
    max_attempts = 10000

    if avg_exact==0:
        threshold = 1.0
    else:
        threshold = 0.01 * avg_exact

    print("Generating power query log...")
    # print("The threshold is ", threshold)
    query_log = []
    while len(query_log) < num_queries and attempts < max_attempts:
        q = generate_bounded_random_query(data, dimensions)
        exact_result = exact_sum(agg_col, q, data)
        estimate = sample_sum(agg_col, q, sample, full_data_size)
        if exact_result > threshold and estimate > 0:  # the result will propbably be 0, make threshold to 1 to ensure valuable result.
            error = exact_result - estimate
            query_log.append({'query': q, 'exact': exact_result, 
                            'estimate': estimate, 'error': error})
        attempts += 1

    if avg_exact==0 :
        avg_exact = np.mean([temp_query['exact'] for temp_query in query_log])
        print(f"Average exact sum: {avg_exact=}")
        print(f"Generated {len(query_log)} queries in {attempts} attempts\n")
        return avg_exact, query_log
    else:
        print(f"Generated {len(query_log)} queries in {attempts} attempts\n")
        return query_log