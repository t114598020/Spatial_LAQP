from sklearn.preprocessing import StandardScaler
import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances

# Diversification (to 800)
def diversify_query_log(log, dimensions, k=800):
    features = []
    for entry in log:
        vec = [entry['query'][dim][i] for dim in dimensions for i in range(2)]
        vec.append(entry['error'])
        features.append(vec)
    features = np.array(features)
    scaler_div = StandardScaler().fit_transform(features)
    selected_indices = [random.randint(0, len(features)-1)]
    selected_features = scaler_div[selected_indices]
    while len(selected_indices) < k:
        dists = euclidean_distances(selected_features, scaler_div)
        min_dists = dists.min(axis=0)
        next_idx = np.argmax(min_dists)
        selected_indices.append(next_idx)
        selected_features = scaler_div[selected_indices]
    
    diversified_log = [log[i] for i in selected_indices]
    print(f"Diversified query log: selected {len(diversified_log)} / {len(log)} entries")
    return [log[i] for i in selected_indices]