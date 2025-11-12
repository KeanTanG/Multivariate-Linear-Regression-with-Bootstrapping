# Bootstrapped Multivariate Linear Regression in Python

A Python implementation of multivariate linear regression with bootstrapping using residual shuffling. This package provides functions for fitting multivariate ordinary least squares (OLS) regression and bootstrapped regression with confidence intervals and p-values.

## Features

- **Multivariate OLS Regression**: Fit multiple response variables simultaneously with a fixed design matrix
- **Bootstrapped Multivariate Regression**: Bootstrap using residual shuffling for more robust confidence intervals
- **Comprehensive Statistics**: Calculate coefficients, standard errors, confidence intervals, p-values, and R-squared
- **Synthetic Data Generation**: Generate test datasets for experimentation
- **Visualization**: Plot coefficient estimates with confidence intervals and bootstrap distributions

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from multivariate_regression import (
    generate_synthetic_data,
    multivariate_ols,
    bootstrapped_multivariate_regression
)

# Generate synthetic data
X, Y, true_coefficients = generate_synthetic_data(
    n_samples=100,
    n_features=5,
    n_responses=3,
    noise_std=1.0,
    random_state=42
)

# Fit multivariate OLS
ols_results = multivariate_ols(X, Y)

# Fit bootstrapped regression
bootstrap_results = bootstrapped_multivariate_regression(
    X, Y,
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=42
)

# Access results
print("OLS Coefficients:", ols_results['coefficients'])
print("Bootstrap Coefficients:", bootstrap_results['coefficients'])
print("Confidence Intervals:", bootstrap_results['confidence_intervals'])
print("P-values:", bootstrap_results['p_values'])
```

## Usage

### Running the Main Demo

```bash
python multivariate_regression.py
```

This will run a complete demonstration including:
- Synthetic data generation
- Multivariate OLS fitting
- Bootstrapped regression fitting
- Comparison of both methods

### Running Examples

```bash
python example.py
```

This includes multiple examples:
1. **Basic Usage**: Simple demonstration of both methods
2. **With Visualization**: Generates plots comparing coefficient estimates
3. **Small Sample Size**: Shows differences between methods with limited data

## Functions

### `multivariate_ols(X, Y)`
Performs multivariate ordinary least squares regression.

**Returns:**
- `coefficients`: Estimated coefficients
- `residuals`: Regression residuals
- `fitted_values`: Fitted values
- `std_errors`: Standard errors of coefficients
- `confidence_intervals`: 95% confidence intervals
- `p_values`: P-values for hypothesis tests
- `r_squared`: R-squared for each response

### `bootstrapped_multivariate_regression(X, Y, n_bootstrap=1000, confidence_level=0.95, random_state=None)`
Performs bootstrapped multivariate regression using residual shuffling.

**Parameters:**
- `X`: Design matrix (n_samples, n_features)
- `Y`: Response matrix (n_samples, n_responses)
- `n_bootstrap`: Number of bootstrap iterations (default: 1000)
- `confidence_level`: Confidence level for intervals (default: 0.95)
- `random_state`: Random seed for reproducibility

**Returns:**
- `coefficients`: Mean bootstrapped coefficients
- `confidence_intervals`: Bootstrap confidence intervals
- `p_values`: Bootstrap p-values
- `bootstrap_coefficients`: All bootstrap coefficient samples
- `std_errors`: Bootstrap standard errors
- `r_squared`: R-squared from original fit

### `generate_synthetic_data(n_samples=100, n_features=5, n_responses=3, noise_std=1.0, random_state=None)`
Generates synthetic data for testing.

**Parameters:**
- `n_samples`: Number of samples
- `n_features`: Number of features (including intercept)
- `n_responses`: Number of response variables
- `noise_std`: Standard deviation of Gaussian noise
- `random_state`: Random seed for reproducibility

**Returns:**
- `X`: Design matrix
- `Y`: Response matrix
- `true_coefficients`: True coefficients used for generation

## Methodology

### Bootstrapped Regression with Residual Shuffling

1. Fit initial OLS model to obtain coefficients and residuals
2. For each bootstrap iteration:
   - Resample residuals with replacement
   - Add resampled residuals to fitted values to create bootstrapped responses
   - Fit OLS on bootstrapped data
3. Calculate statistics from bootstrap distribution:
   - Mean coefficients
   - Percentile-based confidence intervals
   - P-values based on proportion of coefficients crossing zero

This method maintains the fixed design matrix (X) while resampling residuals, which is appropriate when the predictor variables are considered fixed rather than random.

## Requirements

- Python >= 3.7
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0 (for visualization in examples)

## License

See LICENSE file for details.