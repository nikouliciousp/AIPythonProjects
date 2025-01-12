"""
linear_regression.py

This script demonstrates how to perform linear regression using Python,

Author: Peris Nik
"""

import matplotlib.pyplot as plt


def compute_linear_regression(x, y):
    """
    Perform simple linear regression without NumPy.
    Computes the slope (β1) and intercept (β0) of the linear regression line.
    """
    # Compute the means of x and y
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    # Calculate the numerator and denominator for the slope (β1)
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)
    slope = numerator / denominator

    # Calculate the intercept (β0)
    intercept = y_mean - slope * x_mean

    return intercept, slope


def predict(x, intercept, slope):
    """
    Generate predictions based on the linear model.
    Uses the computed slope and intercept to calculate predicted y values for a given x.
    """
    return [intercept + slope * xi for xi in x]


def main():
    """
    Main function to demonstrate the use of linear regression functions.
    Computes the model, displays coefficients, predicts values, and creates a diagram.
    """
    # Sample data (x, y)
    x = [1, 2, 3, 4, 5]  # Independent variable
    y = [3, 4, 2, 5, 6]  # Dependent variable

    # Compute coefficients for the linear regression model
    intercept, slope = compute_linear_regression(x, y)

    # Display the intercept and slope
    print(f"Intercept (β0): {intercept:.2f}")
    print(f"Slope (β1): {slope:.2f}")

    # Make predictions for the given x values
    y_pred = predict(x, intercept, slope)

    # Display the predicted values
    print("Predicted values:")
    for xi, yi in zip(x, y_pred):
        print(f"x = {xi}, Predicted y = {yi:.2f}")

    # Create a diagram
    plt.scatter(x, y, color="blue", label="Actual data")
    plt.plot(x, y_pred, color="red", label="Regression line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
