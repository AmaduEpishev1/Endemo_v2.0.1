"""
This module contains the functions for forecast
"""
import numpy as np

def _predict_constant_last(coef, x_values) -> float:
    """
    Predict using the last constant value method.

    :return: The last known value (constant prediction).
    """
    if coef is None:
        raise ValueError("Constant prediction cannot be performed. Offset (constant value) is not set.")
    return coef.coefficients[0]

def _predict_constant_mean(coef, x_values) -> float:
    """
    Predict using the constant mean value method.

    :return: The mean value (constant mean prediction).
    """
    if coef.coefficients is None:
        raise ValueError("Mean prediction cannot be performed. Offset (constant value) is not set.")
    return coef.coefficients[0]

def _predict_lin(coef, x_values: list) -> float:
    """
    Perform prediction using the provided coefficients and independent variables.

    :param x_values: List of independent variable values.
    :return: Predicted value.
    """
    x_values = [float(x) for x in x_values]

    coefficients_x = coef.coefficients[1:]  # Exclude the intercept (`k0`)
    offset = coef.coefficients[0]  # Retrieve the intercept (`k0`)

    if len(coefficients_x) != len(x_values):
        raise ValueError("Mismatch between the number of coefficients and independent variables.")

    return offset + sum(c * x for c, x in zip(coefficients_x, x_values))


def calculate_const_mult_div(coef,x_values: list) -> float:
    """
    Calculates the result of the formula K0 * (X1 / X2).

    :param coef: Coefficient value (K0).
    :param values: A list containing two values [X1, X2].
    :return: The calculated result or raises an exception for invalid inputs.
    """
    if len(x_values) != 2:
        raise ValueError("Values must be a list of exactly two elements: [X1, X2].")
    offset = coef.coefficients[0]  # Retrieve the intercept (`k0`)
    X1, X2 = x_values
    if X2 == 0:
        raise ValueError("X2 cannot be zero to avoid division by zero.")

    return offset * (X1 / X2)

def calculate_lin_mult_div(coef, x_values: list) -> float:
    """
    Perform calculation based on the formula: (k0 + k1 * X1) * X2 / X3.

    :param coef: Coefficient object containing intercept and slope.
    :param x_values: List of independent variable values [X1, X2, X3].
    :return: Calculated result.
    """
    if len(x_values) != 3:
        raise ValueError("x_values must contain exactly three elements: [X1, X2, X3].")

    X1, X2, X3 = map(float, x_values)  # Ensure all inputs are floats

    if X3 == 0:
        raise ValueError("X3 cannot be zero to avoid division by zero.")

    coefficients = coef.coefficients  # Extract only k0 and k1
    if len(coefficients) != 2:
        raise ValueError("LIN_MULT_DIV requires exactly two coefficients: k0 and k1.")

    k0, k1 = coefficients
    return ((k0 + k1 * X1) * X2) / X3

def exp_mult_div(coef, x_values: list) -> float:
    """
    Perform calculation based on the formula: k0 + k1 * ((1 + k2 / 100) ^ (X1 - k3)) * X2 / X3.

    :param coef: Coefficient object containing the coefficients [k0, k1, k2, k3].
    :param x_values: List of independent variable values [X1, X2, X3].
    :return: Calculated result.
    """
    if len(x_values) != 3:
        raise ValueError("x_values must contain exactly three elements: [X1, X2, X3].")

    X1, X2, X3 = map(float, x_values)  # Ensure all inputs are floats

    if X3 == 0:
        raise ValueError("X3 cannot be zero to avoid division by zero.")

    coefficients = coef.coefficients  # Extract only k0, k1, k2, k3
    if len(coefficients) != 4:
        raise ValueError("EXP_MULT_DIV requires exactly four coefficients: k0, k1, k2, k3.")

    k0, k1, k2, k3 = coefficients

    growth_factor = (1 + k2 / 100) ** (X1 - k3)  # Calculate the growth factor
    return k0 + k1 * growth_factor * (X2 / X3)

def exponential_multivariable(coef, x_values: list) -> float:
    """
    Perform exponential multivariable calculation.
    Formula: y = k0 + kn * exp(k1 * X1 + k2 * X2 + ... + kn-1 * Xn-1)

    :param coef: Coefficient object containing the coefficients [k0, kn, k1, k2, ..., kn-1].
    :param x_values: List of independent variable values [X1, X2, ..., Xn-1].
    :return: Calculated result.
    """
    coefficients = coef.coefficients  # Retrieve all coefficients
    if len(coefficients) < 3:
        raise ValueError("Exponential multivariable requires at least three coefficients: k0, kn, and k1.")

    k0, kn = coefficients[:2]  # Intercept (k0) and exponential scaling factor (kn)
    kn_minus_1 = coefficients[2:]  # Remaining coefficients [k1, k2, ..., kn-1]

    if len(kn_minus_1) != len(x_values):
        raise ValueError("Mismatch between the number of coefficients and independent variables.")

    # Compute the linear combination in the exponent
    exponent = sum(k * x for k, x in zip(kn_minus_1, x_values))

    # Compute the final result
    return k0 + kn * np.exp(exponent)

def multivariable_interpolation(point, points):
    """
    Perform linear interpolation in n-dimensional space.

    Parameters:
        point: A tuple of coordinates (x, y, z, ...) where interpolation is desired.
        points: A list of tuples [(coord1, coord2, ..., value), ...]
                where coord1, coord2, ... are the coordinates, and value is the function value at that point.

    Returns:
        Interpolated value at the given point.
    """
    n = len(point)  # Dimensionality of the input point

    if len(points) == 1:
        # Base case: Only one point, return its value
        return points[0][-1]

    # Group points by the first coordinate
    grouped = {}
    for p in points:
        key = p[0]
        grouped.setdefault(key, []).append(p[1:])

    # Sort the keys and check bounds
    keys = sorted(grouped.keys())
    if not (keys[0] <= point[0] <= keys[-1]):
        raise ValueError(f"Point {point} is out of bounds in dimension 0")

    # Interpolate in the first dimension
    lower_key = max(k for k in keys if k <= point[0])
    upper_key = min(k for k in keys if k >= point[0])

    if lower_key == upper_key:
        # No need to interpolate, recurse in remaining dimensions
        return multivariable_interpolation(point[1:], grouped[lower_key])

    # Values at the lower and upper keys
    lower_value = multivariable_interpolation(point[1:], grouped[lower_key])
    upper_value = multivariable_interpolation(point[1:], grouped[upper_key])

    # Linear interpolation in the current dimension
    weight = (point[0] - lower_key) / (upper_key - lower_key)
    return (1 - weight) * lower_value + weight * upper_value


