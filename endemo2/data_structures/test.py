import pandas as pd
import numpy as np
# Create the list of known points

row_future_data_values = [10, 20, 30, 40]  # Function values
demand_driver_array = np.array([
    [1, 2,5],  # Coordinates for f(1, 2)
    [1, 4,12],  # Coordinates for f(1, 4)
    [3, 2,15],  # Coordinates for f(3, 2)
    [3, 4,16],  # Coordinates for f(3, 4)
])

known_points = [
    tuple(coord) + (value,)
    for coord, value in zip(demand_driver_array, row_future_data_values)
]

# Example Outputs
print("Known points for interpolation:")
print(known_points)
# for point in known_points:
#     print(point)
import numpy as np

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

point = (1, 2, 10)
interpolated_value = multivariable_interpolation(point, known_points)
print(f"Interpolated value at {point}: {interpolated_value}")
print(type(interpolated_value))