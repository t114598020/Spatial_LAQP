import numpy as np

# Function to extract features (flattened ranges + error)
def extract_features(query_log, dimensions):
    features = []
    for entry in query_log:
        vec = []
        for dim in dimensions:
            lower, upper = entry['query'][dim]
            vec.extend([lower, upper])
        vec.append(entry['error'])
        features.append(vec)
    return np.array(features)