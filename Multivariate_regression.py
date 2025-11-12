"""
Multivariate Linear Regression with Bootstrapping
This module provides functions for:
- Multivariate OLS regression (multiple responses)
- Bootstrapped multivariate regression with residual shuffling
- Synthetic data generation
- Confidence intervals and p-values
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional


def multivariate_ols(X: np.ndarray, Y: np.ndarray) -> Dict:
    """
    Perform multivariate ordinary least squares regression.
    
    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    Y : np.ndarray
        Response matrix of shape (n_samples, n_responses)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'coefficients': Estimated coefficients (n_features, n_responses)
        - 'residuals': Residuals (n_samples, n_responses)
        - 'fitted_values': Fitted values (n_samples, n_responses)
        - 'std_errors': Standard errors of coefficients (n_features, n_responses)
        - 'confidence_intervals': 95% confidence intervals (n_features, n_responses, 2)
        - 'p_values': P-values for coefficients (n_features, n_responses)
        - 'r_squared': R-squared for each response (n_responses,)
    """
    n_samples, n_features = X.shape
    n_responses = Y.shape[1] if Y.ndim > 1 else 1

    # Ensure Y is 2D
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # Calculate coefficients using OLS formula: beta = (X'X)^(-1) X'Y
    XtX = X.T @ X
    XtY = X.T @ Y
    coefficients = np.linalg.solve(XtX, XtY)

    # Calculate fitted values and residuals
    fitted_values = X @ coefficients
    residuals = Y - fitted_values

    # Calculate residual sum of squares
    rss = np.sum(residuals**2, axis=0)

    # Degrees of freedom
    df_residual = n_samples - n_features

    # Estimate variance of residuals
    sigma_squared = rss / df_residual

    # Covariance matrix of coefficients for each response
    XtX_inv = np.linalg.inv(XtX)

    # Standard errors
    std_errors = np.zeros((n_features, n_responses))
    for j in range(n_responses):
        var_beta = sigma_squared[j] * XtX_inv
        std_errors[:, j] = np.sqrt(np.diag(var_beta))

    # T-statistics
    t_stats = coefficients / std_errors

    # P-values (two-tailed test)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_residual))

    # 95% confidence intervals
    t_critical = stats.t.ppf(0.975, df_residual)
    ci_lower = coefficients - t_critical * std_errors
    ci_upper = coefficients + t_critical * std_errors
    confidence_intervals = np.stack([ci_lower, ci_upper], axis=-1)

    # Calculate R-squared
    tss = np.sum((Y - np.mean(Y, axis=0))**2, axis=0)
    r_squared = 1 - rss / tss

    return {
        'coefficients': coefficients,
        'residuals': residuals,
        'fitted_values': fitted_values,
        'std_errors': std_errors,
        'confidence_intervals': confidence_intervals,
        'p_values': p_values,
        'r_squared': r_squared
    }


def bootstrapped_multivariate_regression(
    X: np.ndarray,
    Y: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Dict:
    """
    Perform bootstrapped multivariate linear regression using residual shuffling.
    
    This method:
    1. Fits OLS to get initial coefficients and residuals
    2. Bootstraps by resampling residuals and adding to fitted values
    3. Refits model on bootstrapped data
    4. Computes confidence intervals from bootstrap distribution
    
    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    Y : np.ndarray
        Response matrix of shape (n_samples, n_responses)
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
    confidence_level : float, default=0.95
        Confidence level for confidence intervals
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'coefficients': Mean bootstrapped coefficients (n_features, n_responses)
        - 'confidence_intervals': Bootstrap confidence intervals (n_features, n_responses, 2)
        - 'p_values': Bootstrap p-values (n_features, n_responses)
        - 'bootstrap_coefficients': All bootstrap coefficients (n_bootstrap, n_features, n_responses)
        - 'std_errors': Bootstrap standard errors (n_features, n_responses)
        - 'r_squared': R-squared from original fit (n_responses,)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    n_responses = Y.shape[1] if Y.ndim > 1 else 1

    # Ensure Y is 2D
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # Initial OLS fit
    initial_fit = multivariate_ols(X, Y)
    fitted_values = initial_fit['fitted_values']
    residuals = initial_fit['residuals']

    # Bootstrap
    bootstrap_coefficients = np.zeros((n_bootstrap, n_features, n_responses))

    for i in range(n_bootstrap):
        # Resample residuals (with replacement)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        resampled_residuals = residuals[bootstrap_indices]

        # Create bootstrapped Y by adding resampled residuals to fitted values
        Y_bootstrap = fitted_values + resampled_residuals

        # Fit OLS on bootstrapped data
        bootstrap_fit = multivariate_ols(X, Y_bootstrap)
        bootstrap_coefficients[i] = bootstrap_fit['coefficients']

    # Calculate statistics from bootstrap distribution
    mean_coefficients = np.mean(bootstrap_coefficients, axis=0)
    std_errors = np.std(bootstrap_coefficients, axis=0, ddof=1)

    # Confidence intervals using percentile method
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    ci_lower = np.percentile(bootstrap_coefficients, lower_percentile, axis=0)
    ci_upper = np.percentile(bootstrap_coefficients, upper_percentile, axis=0)
    confidence_intervals = np.stack([ci_lower, ci_upper], axis=-1)

    # P-values: proportion of bootstrap samples where coefficient crosses zero
    # Two-tailed test
    p_values = np.zeros((n_features, n_responses))
    for j in range(n_features):
        for k in range(n_responses):
            bootstrap_coefs = bootstrap_coefficients[:, j, k]
            # Proportion of bootstrap coefficients with opposite sign of mean
            prop_opposite = np.mean(bootstrap_coefs * mean_coefficients[j, k] < 0)
            # Two-tailed p-value
            p_values[j, k] = 2 * min(prop_opposite, 1 - prop_opposite)

    return {
        'coefficients': mean_coefficients,
        'confidence_intervals': confidence_intervals,
        'p_values': p_values,
        'bootstrap_coefficients': bootstrap_coefficients,
        'std_errors': std_errors,
        'r_squared': initial_fit['r_squared']
    }


def generate_synthetic_data(
    n_samples: int = 100,
    n_features: int = 5,
    n_responses: int = 3,
    noise_std: float = 1.0,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data for multivariate linear regression.
    
    Parameters:
    -----------
    n_samples : int, default=100
        Number of samples
    n_features : int, default=5
        Number of features (including intercept)
    n_responses : int, default=3
        Number of response variables
    noise_std : float, default=1.0
        Standard deviation of Gaussian noise
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    tuple : (X, Y, true_coefficients)
        - X: Design matrix (n_samples, n_features)
        - Y: Response matrix (n_samples, n_responses)
        - true_coefficients: True coefficients used to generate Y (n_features, n_responses)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate design matrix with intercept
    X = np.ones((n_samples, n_features))
    X[:, 1:] = np.random.randn(n_samples, n_features - 1)

    # Generate true coefficients
    true_coefficients = np.random.randn(n_features, n_responses)

    # Generate responses with noise
    Y = X @ true_coefficients + noise_std * np.random.randn(n_samples, n_responses)

    return X, Y, true_coefficients


def print_results(results: Dict, method_name: str = "OLS"):
    """
    Print regression results in a formatted way.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from multivariate_ols or bootstrapped_multivariate_regression
    method_name : str
        Name of the method for display
    """
    print(f"\n{'='*60}")
    print(f"{method_name} Results")
    print(f"{'='*60}\n")

    coefficients = results['coefficients']
    n_features, n_responses = coefficients.shape

    print(f"Coefficients:")
    print(f"{'-'*60}")
    for i in range(n_features):
        print(f"Feature {i}:", end="")
        for j in range(n_responses):
            print(f"  Response {j}: {coefficients[i, j]:.4f}", end="")
        print()

    print(f"\n{'='*60}")
    print(f"Confidence Intervals (95%):")
    print(f"{'-'*60}")
    ci = results['confidence_intervals']
    for i in range(n_features):
        print(f"Feature {i}:")
        for j in range(n_responses):
            print(f"  Response {j}: [{ci[i, j, 0]:.4f}, {ci[i, j, 1]:.4f}]")

    print(f"\n{'='*60}")
    print(f"P-values:")
    print(f"{'-'*60}")
    p_vals = results['p_values']
    for i in range(n_features):
        print(f"Feature {i}:", end="")
        for j in range(n_responses):
            significance = "***" if p_vals[i, j] < 0.001 else "**" if p_vals[i, j] < 0.01 else "*" if p_vals[i, j] < 0.05 else ""
            print(f"  Response {j}: {p_vals[i, j]:.4f} {significance}", end="")
        print()

    print(f"\n{'='*60}")
    print(f"R-squared:")
    print(f"{'-'*60}")
    r_squared = results['r_squared']
    for j in range(n_responses):
        print(f"Response {j}: {r_squared[j]:.4f}")
    print(f"{'='*60}\n")


def main():
    """
    Main function demonstrating multivariate OLS and bootstrapped regression.
    """
    print("Multivariate Linear Regression with Bootstrap")
    print("=" * 60)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    n_samples = 100
    n_features = 5
    n_responses = 3

    X, Y, true_coefficients = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        n_responses=n_responses,
        noise_std=1.0,
        random_state=42
    )

    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    print(f"\nTrue coefficients:")
    print(true_coefficients)

    # Perform multivariate OLS
    print("\n" + "=" * 60)
    print("Performing Multivariate OLS...")
    ols_results = multivariate_ols(X, Y)
    print_results(ols_results, "Multivariate OLS")

    # Perform bootstrapped multivariate regression
    print("\n" + "=" * 60)
    print("Performing Bootstrapped Multivariate Regression...")
    print("(This may take a moment...)")
    bootstrap_results = bootstrapped_multivariate_regression(
        X, Y,
        n_bootstrap=1000,
        confidence_level=0.95,
        random_state=42
    )
    print_results(bootstrap_results, "Bootstrapped Multivariate Regression")

    # Compare methods
    print("\n" + "=" * 60)
    print("Comparison of Methods:")
    print("=" * 60)
    print("\nDifference in estimated coefficients:")
    coef_diff = np.abs(ols_results['coefficients'] - bootstrap_results['coefficients'])
    print(f"Max difference: {np.max(coef_diff):.6f}")
    print(f"Mean difference: {np.mean(coef_diff):.6f}")

    print("\nDifference in standard errors:")
    se_diff = np.abs(ols_results['std_errors'] - bootstrap_results['std_errors'])
    print(f"Max difference: {np.max(se_diff):.6f}")
    print(f"Mean difference: {np.mean(se_diff):.6f}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()