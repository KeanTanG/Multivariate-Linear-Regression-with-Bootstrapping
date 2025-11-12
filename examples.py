"""
Example usage of multivariate linear regression with bootstrapping.
This example demonstrates:
1. Generating synthetic data
2. Fitting multivariate OLS
3. Fitting bootstrapped multivariate regression
4. Comparing results between methods
5. Visualizing coefficient estimates and confidence intervals
"""

import numpy as np
import matplotlib.pyplot as plt
from Multivariate_regression import (
    generate_synthetic_data,
    multivariate_ols,
    bootstrapped_multivariate_regression,
    print_results
)


def plot_coefficients_comparison(ols_results, bootstrap_results, true_coefficients):
    """
    Plot comparison of coefficient estimates between OLS and Bootstrap methods.
    
    Parameters:
    -----------
    ols_results : dict
        Results from multivariate_ols
    bootstrap_results : dict
        Results from bootstrapped_multivariate_regression
    true_coefficients : np.ndarray
        True coefficients used to generate data
    """
    n_features, n_responses = true_coefficients.shape

    fig, axes = plt.subplots(n_responses, 1, figsize=(12, 4 * n_responses))
    if n_responses == 1:
        axes = [axes]

    for j in range(n_responses):
        ax = axes[j]
        x_pos = np.arange(n_features)
        width = 0.25

        # Plot true coefficients
        ax.bar(x_pos - width, true_coefficients[:, j], width, 
               label='True', alpha=0.8, color='green')

        # Plot OLS estimates with error bars
        ols_coef = ols_results['coefficients'][:, j]
        ols_ci = ols_results['confidence_intervals'][:, j]
        ols_err = np.abs(ols_ci - ols_coef[:, np.newaxis]).T
        ax.bar(x_pos, ols_coef, width, 
               label='OLS', alpha=0.8, color='blue')
        ax.errorbar(x_pos, ols_coef, yerr=ols_err, fmt='none', 
                   color='blue', capsize=5, alpha=0.6)

        # Plot Bootstrap estimates with error bars
        boot_coef = bootstrap_results['coefficients'][:, j]
        boot_ci = bootstrap_results['confidence_intervals'][:, j]
        boot_err = np.abs(boot_ci - boot_coef[:, np.newaxis]).T
        ax.bar(x_pos + width, boot_coef, width, 
               label='Bootstrap', alpha=0.8, color='red')
        ax.errorbar(x_pos + width, boot_coef, yerr=boot_err, fmt='none', 
                   color='red', capsize=5, alpha=0.6)

        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Coefficient Value')
        ax.set_title(f'Response {j}: Coefficient Estimates with 95% CI')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'β{i}' for i in range(n_features)])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    return fig


def plot_bootstrap_distribution(bootstrap_results, feature_idx=1, response_idx=0):
    """
    Plot the bootstrap distribution for a specific coefficient.
    
    Parameters:
    -----------
    bootstrap_results : dict
        Results from bootstrapped_multivariate_regression
    feature_idx : int
        Index of feature to plot
    response_idx : int
        Index of response to plot
    """
    boot_coefs = bootstrap_results['bootstrap_coefficients'][:, feature_idx, response_idx]
    mean_coef = bootstrap_results['coefficients'][feature_idx, response_idx]
    ci = bootstrap_results['confidence_intervals'][feature_idx, response_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    ax.hist(boot_coefs, bins=50, density=True, alpha=0.7, color='skyblue', 
            edgecolor='black', label='Bootstrap distribution')

    # Plot mean
    ax.axvline(mean_coef, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_coef:.4f}')

    # Plot confidence interval
    ax.axvline(ci[0], color='orange', linestyle='--', linewidth=2, 
               label=f'95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]')
    ax.axvline(ci[1], color='orange', linestyle='--', linewidth=2)

    # Fill CI region
    ax.axvspan(ci[0], ci[1], alpha=0.2, color='orange')

    ax.set_xlabel('Coefficient Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Bootstrap Distribution: Feature {feature_idx}, Response {response_idx}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def example_basic_usage():
    """
    Basic example of using multivariate regression functions.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)

    # Generate data
    X, Y, true_coefficients = generate_synthetic_data(
        n_samples=100,
        n_features=4,
        n_responses=2,
        noise_std=0.5,
        random_state=123
    )

    print(f"\nData generated:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Responses: {Y.shape[1]}")

    # Fit OLS
    ols_results = multivariate_ols(X, Y)
    print_results(ols_results, "Multivariate OLS")

    # Fit Bootstrap
    bootstrap_results = bootstrapped_multivariate_regression(
        X, Y, n_bootstrap=500, random_state=123
    )
    print_results(bootstrap_results, "Bootstrapped Regression")


def example_with_visualization():
    """
    Example with visualization of results.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: With Visualization")
    print("=" * 70)

    # Generate data with more features
    X, Y, true_coefficients = generate_synthetic_data(
        n_samples=150,
        n_features=5,
        n_responses=3,
        noise_std=1.0,
        random_state=456
    )

    print(f"\nGenerating results for visualization...")

    # Fit both methods
    ols_results = multivariate_ols(X, Y)
    bootstrap_results = bootstrapped_multivariate_regression(
        X, Y, n_bootstrap=1000, random_state=456
    )

    # Create visualizations
    print("\nCreating comparison plot...")
    fig1 = plot_coefficients_comparison(ols_results, bootstrap_results, true_coefficients)
    plt.savefig('/tmp/coefficient_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: /tmp/coefficient_comparison.png")

    print("Creating bootstrap distribution plot...")
    fig2 = plot_bootstrap_distribution(bootstrap_results, feature_idx=1, response_idx=0)
    plt.savefig('/tmp/bootstrap_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved: /tmp/bootstrap_distribution.png")

    plt.close('all')

    print("\nVisualization complete!")


def example_small_sample():
    """
    Example with small sample size to see difference between methods.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Small Sample Size (n=30)")
    print("=" * 70)

    # Generate data with small sample size
    X, Y, true_coefficients = generate_synthetic_data(
        n_samples=30,
        n_features=4,
        n_responses=2,
        noise_std=1.5,
        random_state=789
    )

    print(f"\nSmall dataset: {X.shape[0]} samples")

    # Fit both methods
    ols_results = multivariate_ols(X, Y)
    bootstrap_results = bootstrapped_multivariate_regression(
        X, Y, n_bootstrap=2000, random_state=789
    )

    # Compare confidence interval widths
    print("\n" + "-" * 70)
    print("Confidence Interval Width Comparison:")
    print("-" * 70)

    ols_ci_width = ols_results['confidence_intervals'][:, :, 1] - \
                   ols_results['confidence_intervals'][:, :, 0]
    boot_ci_width = bootstrap_results['confidence_intervals'][:, :, 1] - \
                    bootstrap_results['confidence_intervals'][:, :, 0]

    print(f"\nOLS average CI width: {np.mean(ols_ci_width):.4f}")
    print(f"Bootstrap average CI width: {np.mean(boot_ci_width):.4f}")
    print(f"Difference: {np.mean(boot_ci_width) - np.mean(ols_ci_width):.4f}")

    print("\nDetailed comparison by feature and response:")
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            print(f"  Feature {i}, Response {j}:")
            print(f"    OLS CI width: {ols_ci_width[i, j]:.4f}")
            print(f"    Bootstrap CI width: {boot_ci_width[i, j]:.4f}")


def main():
    """
    Run all examples.
    """
    print("\n" + "=" * 70)
    print("MULTIVARIATE LINEAR REGRESSION EXAMPLES")
    print("=" * 70)

    # Example 1: Basic usage
    example_basic_usage()

    # Example 2: With visualization
    example_with_visualization()

    # Example 3: Small sample comparison
    example_small_sample()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()