import numpy as np
from scipy.optimize import curve_fit
from itertools import combinations_with_replacement #import to the env
from sklearn.linear_model import LinearRegression


def generate_constant_last_coefficient_(X : list, y: list):
    if not y:
        raise ValueError("Data cannot be empty.")

        # Get the last y value from the data
    k0 = y[-1]  # Extract the y value of the last data point
    equation = "y = k0"
    # Return as a list [k0]
    return [k0], equation

def generate_constant_mean_coefficient_(X : list, y: list):
    if not y:
        raise ValueError("y cannot be empty.")
    k0 = np.mean(y)
    equation = "y = k0(t_hist_mean)"
    return [k0], equation


def generate_linear_coefficients_multivariable_sklearn(X: list, y: list):
    """
    Generate multivariate linear regression coefficients using scikit-learn.
    """
    # Convert X and y to numpy arrays
    X = np.array(X)
    y = np.array(y).T

    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Extract coefficients and intercept
    coefficients = model.coef_  # Coefficients for each independent variable
    intercept = model.intercept_  # Intercept (k0)

    # # Construct the equation string
    # terms = [f"{coeff:.7f}*x{i + 1}" for i, coeff in enumerate(coefficients)]
    equation = f"y = k0 + k1*x1+k2*x2+…"
    # Combine coefficients and intercept into a single list
    result = [intercept] + coefficients.tolist()
    return result, equation # Return coefficients and intercept

def calculate_exponential_multivariable_coefficients(X: np.ndarray, y: np.ndarray):
    """
    Generate coefficients for the exponential multivariable regression model:
    y = k0 + kn * exp(k1 * X1 + k2 * X2 + ... + kn-1 * Xn-1)

    :param X: Independent variable data as a 2D NumPy array (shape: [n_samples, n_features]).
    :param y: Dependent variable data as a 1D NumPy array (shape: [n_samples]).
    :return: List of coefficients [k0, kn, k1, k2, ..., kn-1].
    """
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be a 2D array and y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in X and y must match.")

    # Define the model function
    def exponential_model(X, k0, kn, *coeffs):
        linear_combination = np.dot(X, coeffs)  # Compute k1 * X1 + k2 * X2 + ...
        return k0 + kn * np.exp(linear_combination)

    # Initial guess for the parameters
    n_features = X.shape[1]
    initial_guess = [1.0] * (2 + n_features)  # [k0, kn, k1, k2, ..., kn-1]

    # Fit the model using curve_fit
    popt, _ = curve_fit(exponential_model, X, y, p0=initial_guess)
    equation = "y = k0 + kn * exp(k1 * X1 + k2 * X2 + ... + kn-1 * Xn-1)"

    return popt.tolist(), equation

def generate_quadratic_coefficients_multivariable(X : list, y: list):
    # Quadratic Multivariate: y = k1*x1^2 + k2*x2^2 + ... + kn*x1 + km*x2 + k0
    X = np.array(X)
    y = np.array(y)

    # Create quadratic terms and add bias term
    X_quad = np.hstack([X ** 2, X, np.ones((X.shape[0], 1))])
    coeffs, _, _, _ = np.linalg.lstsq(X_quad, y, rcond=None)
    return coeffs  # [k1, k2, ..., kn, km, k0]


def generate_polynomial_coefficients_multivariable(X : list, y: list, degree: int):
    # Polynomial Multivariate: Generalization for any degree

    X = np.array(X)
    y = np.array(y)

    # Generate all polynomial terms up to the given degree
    n_features = X.shape[1]
    terms = []
    for d in range(1, degree + 1):
        terms.extend(combinations_with_replacement(range(n_features), d))

    X_poly = np.ones((X.shape[0], len(terms) + 1))  # Include k0 (bias term)
    for i, term in enumerate(terms):
        X_poly[:, i] = np.prod(X[:, term], axis=1)

    coeffs, _, _, _ = np.linalg.lstsq(X_poly, y, rcond=None)
    return coeffs  # Polynomial coefficients [k1, k2, ..., kn, k0]



def generate_logarithmic_coefficients_multivariable(X: list, y: list):
    """
    Generate multivariate logarithmic regression coefficients.
    Logarithmic Multivariate: y = k1*log(x1) + k2*log(x2) + ... + k0
    :param X: List of independent variables.
    :param y: List of dependent variable values.
    :return: Coefficients and equation string.
    """
    # Convert X and y to numpy arrays
    X = np.array(X).T
    y = np.array(y)

    # Take the natural log of X
    X_log = np.log(X)

    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(X_log, y)

    # Extract coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_

    # Format the equation
    terms = [f"{coeff:.7f}*log(x{i + 1})" for i, coeff in enumerate(coefficients)]
    equation = " + ".join(terms)
    equation += f" + {intercept:.7f}"

    # Combine coefficients and intercept into a single list
    result = [intercept] + coefficients.tolist()
    print(result)
    return f"y = {equation}", result



