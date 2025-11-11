"""
Bootstrapped Multivariate Linear Regression

This module provides functions to perform bootstrapped multivariate linear regression,
including parameter estimation, confidence intervals, p-values, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def generate_synthetic_data(n_samples=100, n_features=3, n_responses = 2,noise_std=1.0, random_seed=42):
    """
    Generate synthetic data for multivariate linear regression.
    
    Parameters:
    -----------
    n_samples : int, default=100
        Number of samples to generate
    n_features : int, default=3
        Number of features (predictors)
    n_responses : int, default=2
        Number of response variables
    noise_std : float, default=1.0
        Standard deviation of the noise
    random_seed : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix
    Y : np.ndarray, shape (n_samples, n_responses)
        Target variable
    true_coefficients : np.ndarray, shape (n_features + 1,)
        True coefficients including intercept
    """
    np.random.seed(random_seed)
    
    # Generate features from a multivariate normal distribution
    X = np.random.randn(n_samples, n_features)
    
    # add intercept term
    X = np.column_stack([np.ones(n_samples), X])

    # Generate true coefficients
    true_coefficients = np.random.randn(n_features + 1, n_responses) * 5
    
    # Generate target variable: y = intercept + X @ slopes + noise
    Y = X @ true_coefficients + np.random.randn(n_samples, n_responses) * noise_std
    
    return X, Y, true_coefficients


def ols_estimate(X_with_intercept, Y):
    """
    Compute Ordinary Least Squares (OLS) parameter estimates.
    
    Parameters:
    -----------
    X_with_intercept : np.ndarray, shape (n_samples, n_features+1)
        Feature matrix with intercept column
    Y : np.ndarray, shape (n_samples, n_responses)
        Target variable
        
    Returns:
    --------
    coefficients : np.ndarray, shape (n_features + 1, n_responses)
        Estimated coefficients including intercept
    residuals : np.ndarray, shape (n_samples, n_responses)
        Residuals (Y - Y_pred)
    """
    
    # Compute OLS estimates: beta = (X'X)^(-1) X'y
    coefficients = np.linalg.solve(X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ Y)
    
    # Compute residuals
    Y_pred = X_with_intercept @ coefficients
    residuals = Y - Y_pred
    
    return coefficients, residuals


def ols_confidence_intervals(X_with_intercept, Y, coefficients, alpha=0.05):
    """
    Compute confidence intervals for OLS estimates using t-distribution.
    
    Parameters:
    -----------
    X_with_intercept : np.ndarray, shape (n_samples, n_features + 1)
        Feature matrix with intercept column
    Y : np.ndarray, shape (n_samples, n_responses)
        Target variable
    coefficients : np.ndarray, shape (n_features + 1, n_responses)
        OLS coefficient estimates
    alpha : float, default=0.05
        Significance level (default gives 95% CI)
        
    Returns:
    --------
    ci_lower : np.ndarray, shape (n_features + 1, n_responses)
        Lower bounds of confidence intervals
    ci_upper : np.ndarray, shape (n_features + 1, n_responses)
        Upper bounds of confidence intervals
    """
    n_samples, n_coefficient = X_with_intercept.shape # n_features + 1
    n_responses = Y.shape[1] 
    
    
    # Compute residuals
    Y_pred = X_with_intercept @ coefficients
    residuals = Y - Y_pred
    
    # Compute standard errors for each response variable
    dof = n_samples - n_coefficient # degrees of freedom

    std_ols=np.zeros((n_coefficient, n_responses))
    for j in range(n_responses):
        sigma2=np.sum(residuals[:,j]**2)/(dof)
        std_ols[:,j]=np.sqrt(np.diag(sigma2*np.linalg.inv(X_with_intercept.T @ X_with_intercept)))
    
    # create conficence intervals 
   
    t_val = stats.t.ppf(1-alpha/2,dof)
    ci_lower = coefficients - t_val * std_ols
    ci_upper = coefficients + t_val * std_ols
    
    return ci_lower, ci_upper


def ols_p_values(X_with_intercept, Y, coefficients):
    """
    Compute p-values for OLS estimates using t-test.
    
    Parameters:
    -----------
    X_with_intercept: np.ndarray, shape (n_samples, n_features + 1)
        Feature matrix with intercept column
    Y : np.ndarray, shape (n_samples, n_responses)
        Target variable
    coefficients : np.ndarray, shape (n_features + 1, n_responses)
        OLS coefficient estimates
        
    Returns:
    --------
    p_values : np.ndarray, shape (n_features + 1, n_responses)
        P-values for each coefficient
    """
    n_samples, n_coefficients = X_with_intercept.shape
    
    dof = n_samples - n_coefficients  # degrees of freedom

    # Residuals
    Y_pred = X_with_intercept @ coefficients
    residuals = Y - Y_pred  # (n_samples, n_responses)

    # (X'X)^(-1) diagonal
    XtX = X_with_intercept.T @ X_with_intercept
    XtX_inv = np.linalg.pinv(XtX)
    inv_diag = np.diag(XtX_inv)  # (n_coefficients,)

    # sigma^2 = RSS / dof
    sigma2 = np.sum(residuals**2, axis=0) / dof  # (n_responses,)

    # SE = sqrt(sigma^2) * sqrt(diag((X'X)^(-1)))
    se = np.sqrt(inv_diag)[:, None] * np.sqrt(sigma2)[None, :]  

    # t-statistics and two-sided p-values
    t_stats = coefficients / se
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), dof)
    return p_values


def bootstrap_regression(X_with_intercept, Y, n_bootstrap=1000, alpha=0.05, random_seed=42):
    """
    Perform bootstrapped multivariate linear regression.
    
    Parameters:
    -----------
    X_with_intercept : np.ndarray, shape (n_samples, n_features + 1)
        Feature matrix
    Y : np.ndarray, shape (n_samples, n_responses)
        Target variable
    n_bootstrap : int, default=1000
        Number of bootstrap samples
    alpha : float, default=0.05
        Significance level for confidence intervals
    random_seed : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    bootstrap_estimates : np.ndarray, shape (n_bootstrap, n_features + 1)
        Bootstrap coefficient estimates
    """
    np.random.seed(random_seed)
    n_samples, n_coefficients = X_with_intercept.shape
    n_responses = Y.shape[1]
    bootstrap_estimates = np.zeros((n_bootstrap, n_coefficients, n_responses))
    coefficients, residuals = ols_estimate(X_with_intercept, Y)


    for i in range(n_bootstrap):
        # Resample residuals with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X_with_intercept
        Y_bootstrap = (X_with_intercept @ coefficients) + residuals[indices]

        # Compute OLS estimates for bootstrap sample
        coefficients_boot, _ = ols_estimate(X_bootstrap, Y_bootstrap)
        bootstrap_estimates[i] = coefficients_boot
    
    return bootstrap_estimates


def bootstrap_confidence_intervals(bootstrap_estimates, alpha=0.05):
    """
    Compute confidence intervals from bootstrap estimates using percentile method.
    
    Parameters:
    -----------
    bootstrap_estimates : np.ndarray, shape (n_bootstrap, n_features + 1, n_responses)
        Bootstrap coefficient estimates
    alpha : float, default=0.05
        Significance level (default gives 95% CI)
        
    Returns:
    --------
    ci_lower : np.ndarray, shape (n_features + 1, n_responses)
        Lower bounds of confidence intervals
    ci_upper : np.ndarray, shape (n_features + 1, n_responses)
        Upper bounds of confidence intervals
    """
    ci_lower = np.percentile(bootstrap_estimates, 100 * (alpha / 2), axis=0)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2), axis=0)
    
    return ci_lower, ci_upper


def bootstrap_p_values(bootstrap_estimates):
    """
    Compute p-values from bootstrap estimates.
    
    The p-value is computed as the proportion of bootstrap estimates
    that are more extreme than 0 (two-tailed test).
    
    Parameters:
    -----------
    bootstrap_estimates : np.ndarray, shape (n_bootstrap, n_features + 1, n_responses)
        Bootstrap coefficient estimates
        
    Returns:
    --------
    p_values : np.ndarray, shape (n_features + 1, n_responses)
        Bootstrap p-values for each coefficient
    """
    p_values = 2 * np.minimum(np.mean(bootstrap_estimates>=0, axis=0), 
                              np.mean(bootstrap_estimates<=0, axis=0))
    
    return p_values


def plot_bootstrap_distributions(bootstrap_estimates, ols_coefficients, 
                                  ci_lower, ci_upper, feature_names=None):
    """
    Plot the distribution of bootstrapped parameters.
    
    Parameters:
    -----------
    bootstrap_estimates : np.ndarray, shape (n_bootstrap, n_features + 1)
        Bootstrap coefficient estimates
    ols_coefficients : np.ndarray, shape (n_features + 1,)
        OLS coefficient estimates
    ci_lower : np.ndarray, shape (n_features + 1,)
        Lower bounds of confidence intervals
    ci_upper : np.ndarray, shape (n_features + 1,)
        Upper bounds of confidence intervals
    feature_names : list of str, optional
        Names of features for plot titles
    """
    n_features = bootstrap_estimates.shape[1]
    
    if feature_names is None:
        feature_names = ['Intercept'] + [f'β{i}' for i in range(1, n_features)]
    
    # Create subplots
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        
        # Plot histogram of bootstrap estimates
        ax.hist(bootstrap_estimates[:, i], bins=50, alpha=0.7, 
                color='skyblue', edgecolor='black', density=True)
        
        # Plot OLS estimate
        ax.axvline(ols_coefficients[i], color='red', linestyle='--', 
                   linewidth=2, label=f'OLS: {ols_coefficients[i]:.3f}')
        
        # Plot confidence intervals
        ax.axvline(ci_lower[i], color='green', linestyle=':', 
                   linewidth=2, label=f'95% CI')
        ax.axvline(ci_upper[i], color='green', linestyle=':', linewidth=2)
        
        # Add shaded region for CI
        y_max = ax.get_ylim()[1]
        ax.fill_betweenx([0, y_max], ci_lower[i], ci_upper[i], 
                         alpha=0.2, color='green')
        
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Density')
        ax.set_title(f'Bootstrap Distribution: {feature_names[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('bootstrap_distributions.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'bootstrap_distributions.png'")
    plt.show()


def main():
    """
    Main function to demonstrate bootstrapped multivariate linear regression.
    """
    print("=" * 80)
    print("Bootstrapped Multivariate Linear Regression")
    print("=" * 80)
    print()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    n_samples = 100
    n_features = 3
    noise_std = 2.0
    X, y, true_coefficients = generate_synthetic_data(
        n_samples=n_samples, 
        n_features=n_features, 
        noise_std=noise_std
    )
    
    print(f"  - Number of samples: {n_samples}")
    print(f"  - Number of features: {n_features}")
    print(f"  - Noise standard deviation: {noise_std}")
    print(f"  - True coefficients: {true_coefficients}")
    print()
    
    # OLS Estimation
    print("-" * 80)
    print("OLS Estimation")
    print("-" * 80)
    ols_coef, residuals = ols_estimate(X, y)
    ols_ci_lower, ols_ci_upper = ols_confidence_intervals(X, y, ols_coef)
    ols_pvals = ols_p_values(X, y, ols_coef)
    
    feature_names = ['Intercept'] + [f'β{i}' for i in range(1, n_features + 1)]
    
    print(f"{'Parameter':<15} {'True Value':<15} {'OLS Est.':<15} "
          f"{'95% CI Lower':<15} {'95% CI Upper':<15} {'P-value':<15}")
    print("-" * 105)
    for i, name in enumerate(feature_names):
        true_val = true_coefficients[i]
        ols_val = ols_coef[i]
        ci_l = ols_ci_lower[i]
        ci_u = ols_ci_upper[i]
        pval = ols_pvals[i]
        print(f"{name:<15} {true_val:<15.4f} {ols_val:<15.4f} "
              f"{ci_l:<15.4f} {ci_u:<15.4f} {pval:<15.6f}")
    print()
    
    # Bootstrap Estimation
    print("-" * 80)
    print("Bootstrap Estimation")
    print("-" * 80)
    n_bootstrap = 1000
    print(f"Performing bootstrap with {n_bootstrap} iterations...")
    bootstrap_est = bootstrap_regression(X, y, n_bootstrap=n_bootstrap)
    boot_ci_lower, boot_ci_upper = bootstrap_confidence_intervals(bootstrap_est)
    boot_pvals = bootstrap_p_values(bootstrap_est, ols_coef)
    
    print(f"{'Parameter':<15} {'True Value':<15} {'Boot. Mean':<15} "
          f"{'95% CI Lower':<15} {'95% CI Upper':<15} {'P-value':<15}")
    print("-" * 105)
    for i, name in enumerate(feature_names):
        true_val = true_coefficients[i]
        boot_mean = np.mean(bootstrap_est[:, i])
        ci_l = boot_ci_lower[i]
        ci_u = boot_ci_upper[i]
        pval = boot_pvals[i]
        print(f"{name:<15} {true_val:<15.4f} {boot_mean:<15.4f} "
              f"{ci_l:<15.4f} {ci_u:<15.4f} {pval:<15.6f}")
    print()
    
    # Comparison
    print("-" * 80)
    print("Comparison: Bootstrap vs OLS")
    print("-" * 80)
    print(f"{'Parameter':<15} {'OLS Est.':<15} {'Boot. Mean':<15} "
          f"{'Difference':<15}")
    print("-" * 60)
    for i, name in enumerate(feature_names):
        ols_val = ols_coef[i]
        boot_mean = np.mean(bootstrap_est[:, i])
        diff = boot_mean - ols_val
        print(f"{name:<15} {ols_val:<15.4f} {boot_mean:<15.4f} {diff:<15.6f}")
    print()
    
    # Bootstrap Statistics
    print("-" * 80)
    print("Bootstrap Statistics")
    print("-" * 80)
    print(f"{'Parameter':<15} {'Mean':<15} {'Std Dev':<15} "
          f"{'Skewness':<15} {'Kurtosis':<15}")
    print("-" * 75)
    for i, name in enumerate(feature_names):
        mean = np.mean(bootstrap_est[:, i])
        std = np.std(bootstrap_est[:, i], ddof=1)
        skew = stats.skew(bootstrap_est[:, i])
        kurt = stats.kurtosis(bootstrap_est[:, i])
        print(f"{name:<15} {mean:<15.4f} {std:<15.4f} {skew:<15.4f} {kurt:<15.4f}")
    print()
    
    # Plot bootstrap distributions
    print("Creating bootstrap distribution plots...")
    plot_bootstrap_distributions(bootstrap_est, ols_coef, 
                                  boot_ci_lower, boot_ci_upper, 
                                  feature_names)
    
    print()
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()