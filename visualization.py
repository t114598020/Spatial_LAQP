import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Function to extract features
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

# Show differences between doing diversification
def diversification_diff(og_query_log, query_log, dimensions):
    pca_scaler = StandardScaler()

    # Extract features for original and diversified
    original_features = extract_features(og_query_log, dimensions)
    diversified_features = extract_features(query_log, dimensions)

    # Standardize
    original_scaled = pca_scaler.fit_transform(original_features)
    diversified_scaled = pca_scaler.transform(diversified_features)

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    original_pca = pca.fit_transform(original_scaled)
    diversified_pca = pca.transform(diversified_scaled)  # Use same PCA

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Before diversification
    axes[0].scatter(original_pca[:, 0], original_pca[:, 1], alpha=0.5)
    axes[0].set_title('Query Distribution Before Diversification')
    axes[0].set_xlabel('PCA Component 1')
    axes[0].set_ylabel('PCA Component 2')

    # After diversification
    axes[1].scatter(diversified_pca[:, 0], diversified_pca[:, 1], alpha=0.5, color='orange')
    axes[1].set_title('Query Distribution After Diversification')
    axes[1].set_xlabel('PCA Component 1')
    axes[1].set_ylabel('PCA Component 2')

    plt.tight_layout()
    plt.show()

# After calculated the metrics, show them using graph
def metrics_comparison(
    laqp_are, aqp_pp_are, sampling_are,
    laqp_mse, aqp_pp_mse, sampling_mse,
):
    methods = ['LAQP', 'AQP++', 'Sampling']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

    # ARE
    are_values = [laqp_are, aqp_pp_are, sampling_are]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    bars1 = ax1.bar(methods, are_values, color=colors, width=0.6)

    ax1.set_ylabel('Average Relative Error (ARE)')
    ax1.set_title('Comparison of ARE (Lower is Better)')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x()+bar.get_width()/2., h,
                 f'{h:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # MSE
    mse_values = [laqp_mse, aqp_pp_mse, sampling_mse]

    fig, ax2 = plt.subplots(figsize=(8, 5))
    bars2 = ax2.bar(methods, mse_values, color=colors, width=0.6)

    ax2.set_ylabel('Mean Squared Error (MSE)')
    ax2.set_title('Comparison of MSE (Lower is Better)')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars2:
        h = bar.get_height()
        txt = f'{h:.1e}' if h > 10000 else f'{h:.2f}'
        ax2.text(bar.get_x()+bar.get_width()/2., h,
                 txt, ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Since the difference between LAQP and AQP++ might be small, compare them through another graph
def metrics_comparison_large(laqp_are, aqp_pp_are, laqp_mse, aqp_pp_mse):
    methods = ['LAQP', 'AQP++']
    are_values = [laqp_are, aqp_pp_are]
    mse_values = [laqp_mse, aqp_pp_mse]
    colors = ['#1f77b4', '#2ca02c']

    # ARE chart
    fig, ax1 = plt.subplots(figsize=(8, 5))
    bars1 = ax1.bar(methods, are_values, color=colors, width=0.6)

    ax1.set_ylabel('Average Relative Error (ARE)')
    ax1.set_title('Comparison of ARE (Lower is Better)')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

    # MSE chart
    fig, ax2 = plt.subplots(figsize=(8, 5))
    bars2 = ax2.bar(methods, mse_values, color=colors, width=0.6)

    ax2.set_ylabel('Mean Squared Error (MSE)')
    ax2.set_title('Comparison of MSE (Lower is Better)')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars2:
        height = bar.get_height()
        ax1_text = f'{height:.1e}' if height > 10000 else f'{height:.2f}'
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                ax1_text, ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

# Show the time cost graph
def query_time_comparison(sampling_ms, aqp_pp_ms, laqp_ms):
    methods = ['Sampling', 'AQP++', 'LAQP']
    times = [sampling_ms*1000, aqp_pp_ms*1000, laqp_ms*1000]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

    # print("Average online query time (per query):")
    # print(f"Sampling: {sampling_ms:.2f} ms")
    # print(f"AQP++:    {aqp_pp_ms:.2f} ms")
    # print(f"LAQP:     {laqp_ms:.2f} ms\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, times, color=colors, width=0.6)

    ax.set_ylabel('Query Time (ms)')
    ax.set_title('Average Online Query Time (Lower is Better)')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h,
                f'{h:.2f} ms',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

# Show the min-max rel diff graph
def min_max_comparison(
    laqp_min_rel, aqp_pp_min_rel, sampling_min_rel,
    laqp_max_rel, aqp_pp_max_rel, sampling_max_rel
):
    methods = ['LAQP', 'AQP++', 'Sampling']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    # Min relative error
    min_values = [laqp_min_rel, aqp_pp_min_rel, sampling_min_rel]

    fig, ax3 = plt.subplots(figsize=(8, 5))
    bars3 = ax3.bar(methods, min_values, color=colors, width=0.6)

    ax3.set_ylabel('Min Relative Error')
    ax3.set_title('Comparison of Minimum Relative Error')
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars3:
        h = bar.get_height()
        ax3.text(bar.get_x()+bar.get_width()/2., h,
                 f'{h:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Max relative error
    max_values = [laqp_max_rel, aqp_pp_max_rel, sampling_max_rel]

    fig, ax4 = plt.subplots(figsize=(8, 5))
    bars4 = ax4.bar(methods, max_values, color=colors, width=0.6)

    ax4.set_ylabel('Max Relative Error')
    ax4.set_title('Comparison of Maximum Relative Error')
    ax4.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars4:
        h = bar.get_height()
        ax4.text(bar.get_x()+bar.get_width()/2., h,
                 f'{h:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def alpha_curve(alphas, err_curve, final_curve):
    plt.figure()
    plt.plot(alphas, err_curve, marker='o')
    plt.xlabel("alpha")
    plt.ylabel("Error mismatch (MSE)")
    plt.title("Replication-oriented objective vs alpha")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(alphas, final_curve, marker='o')
    plt.xlabel("alpha")
    plt.ylabel("Final relative error")
    plt.title("Final estimation error vs alpha")
    plt.grid(True)
    plt.show()
