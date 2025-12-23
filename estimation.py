import numpy as np
from query_calculate import sample_count
from optimization import range_distance

# Optimized (hybrid with best_alpha)
def optimized_laqp_estimate(query_log, new_query, sample, dimensions, model, scaler, full_data_size, best_alpha):
    vec = [new_query[dim][i] for dim in dimensions for i in range(2)]
    vec = np.array([vec])
    scaled = scaler.transform(vec)  # Assuming your scaler
    pred_error = model.predict(scaled)[0]
    
    best_entry = min(query_log, key=lambda e: 
        best_alpha * abs(e['error'] - pred_error) + 
        (1 - best_alpha) * range_distance(dimensions, new_query, e['query']))
    
    sample_new = sample_count(new_query, sample, full_data_size)
    sample_opt = best_entry['estimate']
    opt_est = best_entry['exact'] + (sample_new - sample_opt)
    
    print(f"Optimized LAQP estimate: {opt_est:.2f}")
    print(f"Chosen historical error: {best_entry['error']:.2f}")
    return opt_est, best_entry

def laqp_estimate_with_details(query_log, new_query, sample, dimensions, model, scaler, full_data_size):
    # Flatten and predict error (same as before)
    vec = []
    for dim in dimensions:
        l, u = new_query[dim]
        vec.extend([l, u])
    vec = np.array([vec])
    scaled = scaler.transform(vec)
    pred_error = model.predict(scaled)[0]
    
    # Find the most error-similar historical query
    best_index = -1
    best_error_diff = float('inf')
    best_entry = None

    for idx, entry in enumerate(query_log):
        error_diff = abs(entry['error'] - pred_error)
        if error_diff < best_error_diff:
            best_error_diff = error_diff
            best_index = idx
            best_entry = entry
    
    # Compute estimates
    sample_new = sample_count(new_query, sample, full_data_size)
    sample_opt = best_entry['estimate']
    final_est = best_entry['exact'] + (sample_new - sample_opt)
    
    print(f"Selected optimal query index: {best_index} (out of  {len(query_log)})")
    print(f"Predicted error for new query: {pred_error:.2f}")
    print(f"Chosen historical query error: {best_entry['error']:.2f} (diff: {best_error_diff:.2f})")
    # print("Predicate ranges of chosen query:")
    # for dim, (l, u) in best_entry['query'].items():
    #     print(f"  {dim}: [{l:.2f}, {u:.2f}]")
    # print(f"\nFinal LAQP estimate: {final_est:.2f}")    

    return final_est, best_index, best_entry