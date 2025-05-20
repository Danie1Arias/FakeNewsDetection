import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Fixed parameters
beta = 0.6
N = 1.0     # Normalized total population
S0 = 0.95
I0 = 0.02

# Parameter ranges to test
delta_values = [0.3, 0.4, 0.5, 0.6]
eta_values = [0.4, 0.5, 0.6, 0.7, 0.8]
mu_values = [0.1, 0.2, 0.3, 0.4, 0.5]
gamma_values = [0.01, 0.02, 0.03]

# Store results
results = []

for delta in delta_values:
    for eta in eta_values:
        for mu in mu_values:
            for gamma in gamma_values:
                # Construct Jacobian matrix
                J = np.array([
                    [-beta * I0 / N, 0, -beta * S0 / N, 0],
                    [ beta * I0 / N, -delta, beta * S0 / N, 0],
                    [0, (1 - eta) * delta, -mu, gamma],
                    [0, eta * delta, mu, -gamma]
                ])

                # Eigenvalues and stiffness ratio
                eigvals = np.linalg.eigvals(J)
                eigvals_real = np.real(eigvals)
                abs_eigvals = np.abs(eigvals)
                try:
                    stiffness_ratio = abs_eigvals.max() / abs_eigvals[abs_eigvals > 1e-12].min()
                except:
                    stiffness_ratio = np.inf
                is_stable = np.all(eigvals_real < 0)

                # Store
                results.append({
                    'delta': delta,
                    'eta': eta,
                    'mu': mu,
                    'gamma': gamma,
                    'λ_max_real': eigvals_real.max(),
                    'λ_min_real': eigvals_real.min(),
                    'stiffness_ratio': stiffness_ratio,
                    'stable': is_stable
                })

# Create DataFrame
df = pd.DataFrame(results)

# Filter only stable systems and sort by lowest stiffness
stable_df = df[df['stable'] == True]
top5_best = stable_df.sort_values('stiffness_ratio').head(5).reset_index(drop=True)

# Print results
print("Top 5 most stable and least stiff parameter combinations:")
print(top5_best)

# Plot top 5
plt.figure(figsize=(10, 6))
bars = plt.bar(
    x=top5_best.index,
    height=top5_best['stiffness_ratio'],
    tick_label=[f'η={r["eta"]}, δ={r["delta"]}, μ={r["mu"]}, γ={r["gamma"]}' for _, r in top5_best.iterrows()],
    color='seagreen',
    alpha=0.85
)

# Add annotations
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height * 1.05, f'{height:.1e}', 
             ha='center', va='bottom', fontsize=10, rotation=45)

plt.yscale('log')
plt.ylabel("Stiffness Ratio (log scale)")
plt.xlabel("Top 5 Stable Parameter Combinations")
plt.title("Top 5 Most Stable and Least Stiff SEIR Configurations")
plt.xticks(rotation=30, ha='right')
plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
