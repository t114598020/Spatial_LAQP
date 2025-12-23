from scipy.optimize import minimize_scalar
import numpy as np
from query_calculate import sample_count

# Optimization (Hybrid α)
def range_distance(dimensions, q1, q2):
    vec1 = [q1[dim][i] for dim in dimensions for i in range(2)]
    vec2 = [q2[dim][i] for dim in dimensions for i in range(2)]
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def optimize_alpha(train_query, val_queries, dimensions, scaler, model, sample, full_data_size, bounds=(0,1)):
    """
    Tune alpha for hybrid similarity (paper Section 5.3).
    val_queries: List of {'query': dict, 'exact': float} for tuning.
    Returns best alpha that minimizes average relative error on val set.
    """
    def objective(alpha):
        errors = []
        for vq in val_queries:
            query = vq['query']
            exact = vq['exact']
            
            # Predict error
            vec = [query[dim][i] for dim in dimensions for i in range(2)]
            vec = np.array([vec])
            scaled = scaler.transform(vec)
            pred_error = model.predict(scaled)[0]
            
            # Find best entry with hybrid similarity
            best_entry = min(train_query, key=lambda e: 
                alpha * abs(e['error'] - pred_error) + 
                (1 - alpha) * range_distance(dimensions, query, e['query']))
            
            # LAQP estimate
            sample_new = sample_count(query, sample, full_data_size)
            sample_opt = best_entry['estimate']
            laqp_est = best_entry['exact'] + (sample_new - sample_opt)
            
            # Relative error
            rel_err = abs(laqp_est - exact) / (exact + 1e-6)
            errors.append(rel_err)
        
        return np.mean(errors)
    
    # Optimize alpha
    res = minimize_scalar(objective, bounds=bounds, method='bounded')
    best_alpha = res.x
    print(f"Optimized alpha: {best_alpha:.3f} (MSE on val: {res.fun:.4f})")
    return best_alpha