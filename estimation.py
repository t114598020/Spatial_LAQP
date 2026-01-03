import numpy as np
from query_calculate import sample_count, sample_sum
from optimization import EDis_norm, RDis_norm

# The original LAQP (only depend on error similar)
def laqp_estimate(
    query_log, new_query, sample, dimensions,
    model, scaler, full_data_size, task,
    error_mean, error_std
):
    vec = [new_query[dim][i] for dim in dimensions for i in range(2)]
    vec = np.array([vec])
    pred_error = model.predict(scaler.transform(vec))[0]

    best_entry = min(
        query_log,
        key=lambda e:
            EDis_norm(pred_error, e['error'], error_mean, error_std)
    )

    if task == "COUNT":
        sample_new = sample_count(new_query, sample, full_data_size)
    else:
        sample_new = sample_sum("Global_active_power", new_query, sample, full_data_size)

    est = best_entry['exact'] + (sample_new - best_entry['estimate'])
    est = max(0, est)
    if est == 0:
        print(f"Basic LAQP estimate clamp to 0")
    else:
        print(f"Basic LAQP estimate: {est:.2f}")

    best_error_diff = best_entry['error'] - pred_error
    print(f"Predicted error for new query: {pred_error:.2f}")
    print(f"Chosen historical query error: {best_entry['error']:.2f} (diff: {best_error_diff:.2f})")
    return est, best_entry

# The optimized LAQP (depend on error similar and range similar with alpha control the weight)
def optimized_laqp_estimate(query_log, new_query, sample, dimensions,
                            model, scaler, full_data_size, task, best_alpha,
                            error_mean, error_std, range_mean, range_std):
    vec = [new_query[dim][i] for dim in dimensions for i in range(2)]
    vec = np.array([vec])
    pred_error = model.predict(scaler.transform(vec))[0]
    
    best_entry = min(
        query_log,
        key=lambda e:
            best_alpha * EDis_norm(
                pred_error, e['error'], error_mean, error_std
            )
            + (1 - best_alpha) * RDis_norm(
                dimensions, new_query, e['query'],
                range_mean, range_std
            )
    )
    
    if task == "COUNT":
        sample_new = sample_count(new_query, sample, full_data_size)
    else:
        sample_new = sample_sum("Global_active_power", new_query, sample, full_data_size)

    sample_opt = best_entry['estimate']
    opt_est = best_entry['exact'] + (sample_new - sample_opt)
    opt_est = max(0, opt_est)
    if opt_est == 0:
        print(f"Optimized LAQP estimate clamp to 0")
    else:
        print(f"Optimized LAQP estimate: {opt_est:.2f}")

    best_error_diff = best_entry['error'] - pred_error
    print(f"Predicted error for new query: {pred_error:.2f}")
    print(f"Chosen historical query error: {best_entry['error']:.2f} (diff: {best_error_diff:.2f})")
    return opt_est, best_entry