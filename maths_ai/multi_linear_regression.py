"""
multi_linear_regression.py

This script demonstrates how to perform multiple linear regression using Python,

Author: Peris Nik
"""

import matplotlib.pyplot as plt


def matrix_transpose(matrix):
    return list(map(list, zip(*matrix)))


def matrix_multiply(A, B):
    return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]


def matrix_inverse(matrix):
    # For simplicity, use a manual implementation of the 2x2 matrix inverse
    if len(matrix) == 2 and len(matrix[0]) == 2:
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        if determinant == 0:
            raise ValueError("The matrix is not invertible.")
        return [[matrix[1][1] / determinant, -matrix[0][1] / determinant],
                [-matrix[1][0] / determinant, matrix[0][0] / determinant]]
    else:
        raise ValueError("Matrix inversion is only implemented for 2x2 matrices.")


def add_intercept(X):
    return [[1] + row for row in X]


def perform_multi_linear_regression(data, features, target):
    """
    Perform multiple linear regression on the given dataset.

    Parameters:
    - data (dict): The dataset containing the features and target variable as lists.
    - features (list): A list of column names (keys) corresponding to the independent variables.
    - target (str): The key name of the dependent variable.

    Returns:
    - coefficients (list): The coefficients of the regression model.
    - intercept (float): The intercept of the regression model.
    - metrics (dict): A dictionary containing regression metrics:
                      Mean Squared Error and R-squared.
    """
    # Construct feature and target matrices from the dictionary
    X = [[data[feature][i] for feature in features] for i in range(len(data[features[0]]))]
    y = [[value] for value in data[target]]

    # Add a column of ones to X for the intercept term
    X = add_intercept(X)

    # Compute coefficients using the Normal Equation
    # θ = (X'X)^(-1)X'y
    X_transpose = matrix_transpose(X)
    X_transpose_X = matrix_multiply(X_transpose, X)
    X_transpose_X_inv = matrix_inverse(X_transpose_X)
    X_transpose_y = matrix_multiply(X_transpose, y)
    coefficients = matrix_multiply(X_transpose_X_inv, X_transpose_y)

    # The first element in coefficients is the intercept
    intercept = coefficients[0][0]
    model_coefficients = [c[0] for c in coefficients[1:]]

    # Predict y values using the coefficients
    y_pred = matrix_multiply(X, coefficients)
    y_pred = [val[0] for val in y_pred]

    # Compute regression metrics
    mse = sum((y[i][0] - y_pred[i]) ** 2 for i in range(len(y))) / len(y)
    y_mean = sum(row[0] for row in y) / len(y)
    total_variance = sum((row[0] - y_mean) ** 2 for row in y)
    explained_variance = sum((y_pred[i] - y_mean) ** 2 for i in range(len(y_pred)))
    r2 = explained_variance / total_variance

    metrics = {
        "Mean Squared Error": mse,
        "R-squared": r2
    }

    return model_coefficients, intercept, metrics, y_pred


def main():
    """
    The main entry point of the script. Runs multiple linear regression on example data.
    """
    n_samples = 20

    def generate_noise(stddev):
        import random
        return [random.gauss(0, stddev) for _ in range(n_samples)]

    # Dataset with R^2 = 100%
    data_100 = {
        "feature_1": [i * 0.5 for i in range(1, n_samples + 1)],
        "target": [i for i in range(2, 2 + n_samples)],
    }

    # Dataset with R^2 = 80%
    noise_80 = generate_noise(2)
    data_80 = {
        "feature_1": [i * 0.5 for i in range(1, n_samples + 1)],
        "target": [i for i in range(2, 2 + n_samples)],
    }
    data_80["target"] = [data_80["target"][i] + noise_80[i] for i in range(n_samples)]

    # Dataset with R^2 = 50%
    noise_50 = generate_noise(5)
    data_50 = {
        "feature_1": [i * 0.5 for i in range(1, n_samples + 1)],
        "target": [i for i in range(2, 2 + n_samples)],
    }
    data_50["target"] = [data_50["target"][i] + noise_50[i] for i in range(n_samples)]

    datasets = [
        ("R^2 = 100%", data_100),
        ("R^2 = 80%", data_80),
        ("R^2 = 50%", data_50),
    ]

    for title, data in datasets:
        features = ['feature_1']
        target = 'target'

        # Perform multiple linear regression
        coefficients, intercept, metrics, y_pred = perform_multi_linear_regression(data, features, target)

        # Output results
        print(title)
        print("Model Coefficients:", coefficients)
        print("Intercept:", intercept)
        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print("\n")

        # Create plot
        plt.scatter(data[features[0]], data[target], color='blue', label='Actual Values')
        plt.plot(data[features[0]], y_pred, color='red', label='Predicted Values')
        plt.xlabel(features[0])
        plt.ylabel(target)
        plt.title(f"Regression Results ({title})")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
