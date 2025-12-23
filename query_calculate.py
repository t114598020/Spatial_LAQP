import numpy as np

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
