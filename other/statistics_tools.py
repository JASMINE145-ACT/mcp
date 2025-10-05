import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def perform_one_sample_t_test(df, column_name, hypothesized_mean, alpha=0.05, alternative='greater'):
    # --- Data cleaning and preparation ---
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    
    unit_price = df[column_name].dropna()
    unit_price = pd.to_numeric(unit_price, errors='coerce').dropna()

    if len(unit_price) < 3: # Shapiro test needs at least 3 samples
        print("Not enough data points to perform the analysis.")
        return None

    # --- Basic statistics ---
    n = len(unit_price)
    mean_price = unit_price.mean()
    std_price = unit_price.std(ddof=1) # ddof=1 for sample standard deviation

    # --- One-sample t-test ---
    # H0: μ = hypothesized_mean vs H1: μ > hypothesized_mean (or <, or !=)
    t_statistic = (mean_price - hypothesized_mean) / (std_price / np.sqrt(n))
    
    # Calculate p-value based on the alternative hypothesis
    if alternative == 'greater':
        p_value = 1 - stats.t.cdf(t_statistic, df=n-1)
    elif alternative == 'less':
        p_value = stats.t.cdf(t_statistic, df=n-1)
    elif alternative == 'two-sided':
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n-1))
    else:
        raise ValueError("`alternative` must be 'greater', 'less', or 'two-sided'.")

    # --- 95% Confidence Interval (always two-sided) ---
    ci_margin = stats.t.ppf(1 - alpha/2, df=n-1) * (std_price / np.sqrt(n))
    ci_lower = mean_price - ci_margin
    ci_upper = mean_price + ci_margin

    # --- Normality test (Shapiro-Wilk) ---
    shapiro_stat, shapiro_p = stats.shapiro(unit_price)

    # --- Wilcoxon signed-rank test (robustness check) ---
    # Note: Scipy's wilcoxon requires the 'alternative' parameter to match.
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(unit_price - hypothesized_mean, alternative=alternative)

    # --- Effect size ---
    mean_diff = mean_price - hypothesized_mean

    # --- Compile and print results ---
    result = {
        'n': n,
        'sample_mean': round(mean_price, 2),
        'sample_std': round(std_price, 2),
        'hypothesized_mean': hypothesized_mean,
        't_statistic': round(t_statistic, 4),
        'p_value': round(p_value, 4),
        f'{100*(1-alpha)}%_ci': (round(ci_lower, 2), round(ci_upper, 2)),
        'shapiro_p': round(shapiro_p, 4),
        'wilcoxon_p': round(wilcoxon_p, 4),
        'mean_difference': round(mean_diff, 2),
        'conclusion': f'Reject H0' if p_value < alpha else f'Do not reject H0'
    }

    print(f"=== One-Sample T-Test Results for '{column_name}' ===")
    print(f"H0: μ = {hypothesized_mean} | H1: μ {'>' if alternative=='greater' else '<' if alternative=='less' else '!='} {hypothesized_mean}")
    print("-" * 40)
    print(f"Sample size: {result['n']}")
    print(f"Sample mean: {result['sample_mean']}")
    print(f"Sample std: {result['sample_std']}")
    print(f"T-statistic: {result['t_statistic']}")
    print(f"P-value ({alternative}): {result['p_value']}")
    print(f"{100*(1-alpha)}% CI: {result[f'{100*(1-alpha)}%_ci']}")
    print(f"\nNormality test (Shapiro-Wilk) p-value: {result['shapiro_p']}")
    print(f"Wilcoxon signed-rank test p-value: {result['wilcoxon_p']}")
    print(f"\nMean difference from {hypothesized_mean}: {result['mean_difference']}")
    print(f"Conclusion (alpha={alpha}): {result['conclusion']}")
    print("-" * 40)

    # --- Visualization ---
    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(unit_price, bins=15, edgecolor='black', alpha=0.7, density=True)
    plt.axvline(hypothesized_mean, color='red', linestyle='--', linewidth=2, label=f'H0: μ={hypothesized_mean}')
    plt.axvline(mean_price, color='green', linestyle='-', linewidth=2, label=f'Sample mean={mean_price:.2f}')
    plt.xlabel(f'{column_name}')
    plt.ylabel('Density')
    plt.title(f'Distribution of {column_name}')
    plt.legend()

    # Q-Q Plot
    plt.subplot(1, 2, 2)
    stats.probplot(unit_price, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normality Check)')

    plt.tight_layout()
    plt.savefig('t_test_analysis.png', dpi=100, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    return result
