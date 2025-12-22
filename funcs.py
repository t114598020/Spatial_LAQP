import random
import numpy as np
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.preprocessing import StandardScaler
# scaler_div = StandardScaler()

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

# Function to compute exact SUM for a query
def exact_sum(agg_col, query, df):
    mask = np.ones(len(df), dtype=bool)
    for dim, (lower, upper) in query.items():
        mask &= (df[dim] >= lower) & (df[dim] <= upper)
    return df.loc[mask, agg_col].sum()

# Function for sampling-based approximate SUM (scaled)
def sample_sum(agg_col, query, samp_df, full_size):
    mask = np.ones(len(samp_df), dtype=bool)
    for dim, (lower, upper) in query.items():
        mask &= (samp_df[dim] >= lower) & (samp_df[dim] <= upper)
    subset_sum = samp_df.loc[mask, agg_col].sum()
    scale = full_size / len(samp_df)
    return subset_sum * scale

# Exact COUNT
def exact_count(query, df):
    mask = np.ones(len(df), dtype=bool)
    for dim, (lower, upper) in query.items():
        mask &= (df[dim] >= lower) & (df[dim] <= upper)
    return mask.sum()

# Sampling-based approximate COUNT (scaled)
def sample_count(query, samp_df, full_size):
    mask = np.ones(len(samp_df), dtype=bool)
    for dim, (lower, upper) in query.items():
        mask &= (samp_df[dim] >= lower) & (samp_df[dim] <= upper)
    subset_count = mask.sum()
    scale = full_size / len(samp_df)
    return subset_count * scale