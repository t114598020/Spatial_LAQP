from scipy.optimize import minimize_scalar
from query_calculate import sample_count, sample_sum
import numpy as np

# Do normalization since error and range scale diff
def normalization(query_log, dimensions):
    train_errors = np.array([e['error'] for e in query_log])
    error_mean = train_errors.mean()
    error_std = train_errors.std() + 1e-8

    # Collect all range vectors
    range_vectors = np.array([
        [e['query'][dim][i] for dim in dimensions for i in range(2)]
        for e in query_log
    ])

    range_mean = range_vectors.mean(axis=0)
    range_std = range_vectors.std(axis=0) + 1e-8

    return error_mean, error_std, range_mean, range_std

# Range similar
def RDis_norm(dimensions, q1, q2, mean, std):
    v1 = np.array([q1[dim][i] for dim in dimensions for i in range(2)])
    v2 = np.array([q2[dim][i] for dim in dimensions for i in range(2)])
    
    v1 = (v1 - mean) / std
    v2 = (v2 - mean) / std
    
    return np.mean((v1 - v2) ** 2)

# Error similar
def EDis_norm(pred_error, train_error, mean, std):
    pe = (pred_error - mean) / std
    te = (train_error - mean) / std
    return (pe - te) ** 2

# Do optimization and find the best alpha
def optimize_alpha(train_query, val_queries, dimensions, scaler,
                   error_mean, error_std, range_mean, range_std,
                    model):
    def objective(alpha):
        errors = []
        for vq in val_queries:
            query = vq['query']
            
            vec = [query[dim][i] for dim in dimensions for i in range(2)]
            vec = np.array([vec])
            scaled = scaler.transform(vec)
            pred_error = model.predict(scaled)[0]

            best_entry = min(
                train_query,
                key=lambda e:
                    alpha * EDis_norm(
                        pred_error, e['error'], error_mean, error_std
                    )
                    + (1 - alpha) * RDis_norm(
                        dimensions, query, e['query'],
                        range_mean, range_std
                    )
            )
            # find minimize error diff
            err_diff = pred_error - best_entry['error']
            errors.append(err_diff ** 2)
        
        return np.mean(errors)
    
    res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    best_alpha = res.x
    if best_alpha > 0.999:
        best_alpha = 1.0
    print(f"Optimized alpha: {best_alpha:.4f}")

    return best_alpha

# Show alpha curve
def error_mismatch_objective(
    alpha,
    train_query,
    val_queries,
    dimensions,
    model,
    scaler,
    error_mean,
    error_std,
    range_mean,
    range_std
):
    sq_errors = []

    for vq in val_queries:
        query = vq['query']

        # predict error
        vec = [query[dim][i] for dim in dimensions for i in range(2)]
        vec = np.array([vec])
        pred_error = model.predict(scaler.transform(vec))[0]

        # select best historical query
        best_entry = min(
            train_query,
            key=lambda e:
                alpha * EDis_norm(
                    pred_error, e['error'], error_mean, error_std
                )
                + (1 - alpha) * RDis_norm(
                    dimensions, query, e['query'],
                    range_mean, range_std
                )
        )

        diff = pred_error - best_entry['error']
        sq_errors.append(diff ** 2)

    return np.mean(sq_errors)

# Show final error according to alpha
def final_relative_error(
    alpha,
    train_query,
    eval_queries,
    sample,
    dimensions,
    model,
    scaler,
    full_data_size,
    task,
    error_mean,
    error_std,
    range_mean,
    range_std
):
    rel_errors = []

    for eq in eval_queries:
        query = eq['query']
        exact = eq['exact']

        # predict error
        vec = [query[dim][i] for dim in dimensions for i in range(2)]
        vec = np.array([vec])
        pred_error = model.predict(scaler.transform(vec))[0]

        # select historical query
        best_entry = min(
            train_query,
            key=lambda e:
                alpha * EDis_norm(
                    pred_error, e['error'], error_mean, error_std
                )
                + (1 - alpha) * RDis_norm(
                    dimensions, query, e['query'],
                    range_mean, range_std
                )
        )

        # LAQP estimation
        if task == "COUNT":
            sample_new = sample_count(query, sample, full_data_size)
        else:
            sample_new = sample_sum("Global_active_power", query, sample, full_data_size)

        est = best_entry['exact'] + (sample_new - best_entry['estimate'])
        est = max(0, est)

        rel_err = abs(est - exact) / (exact + 1e-6)
        rel_errors.append(rel_err)

    return np.mean(rel_errors)