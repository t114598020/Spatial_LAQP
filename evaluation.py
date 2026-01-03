import numpy as np

# Show the ARE, MSE and the max-min relative error of SAQP, AQP++, LAQP
def metrics(method, rel, sq):
    are = np.mean(rel)
    mse = np.mean(sq)
    min_rel = np.min(rel)
    max_rel = np.max(rel)

    print(f"\n{method:<6s} Average Relative Error (ARE): {are:.6f}")
    print(f"{method:<6s} Mean Squared Error (MSE):     {mse:.2f}")
    print(f"{method:<6s} Min Relative Error:           {min_rel:.6f}")
    print(f"{method:<6s} Max Relative Error:           {max_rel:.6f}")
    
    return are, mse, min_rel, max_rel
     
# Show the improvement of SAQP, AQP++ and LAQP through ARE
def improvement(sampling_are, aqp_pp_are, laqp_are):
    sampling_improvement = sampling_are / laqp_are if laqp_are > 0 else 0
    print(f"\nImprovement (Sampling ARE / LAQP ARE): {sampling_improvement:.2f}x")
    aqp_pp_improvement = aqp_pp_are / laqp_are if laqp_are > 0 else 0
    print(f"Improvement (AQP++ ARE / LAQP ARE):    {aqp_pp_improvement:.2f}x")

# Calculate abs, rel, sq for metrics 
def general_estimate(estimate, exact):
    method_abs = abs(estimate - exact)
    method_rel = method_abs / exact if exact > 0 else 0
    method_sq = (estimate - exact)**2
    return method_abs, method_rel, method_sq