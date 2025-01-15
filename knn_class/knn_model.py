import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def main():
    # Read the dataset from the CSV file
    data = pd.read_csv("Classified Data.csv", index_col=0)
    print(data.head(15))  # Print the first 15 rows of the dataset for a preview

    # Create a StandardScaler instance for scaling the feature data
    scaler = StandardScaler()

    # Fit the scaler to all features except the target class
    scaler.fit(data.drop("TARGET CLASS", axis=1))

    # Apply the scaling to the features
    scaler_features = scaler.transform(data.drop("TARGET CLASS", axis=1))

    # Create a DataFrame for the scaled features (without column names)
    scaler_data = pd.DataFrame(scaler_features)
    print(scaler_data.head(15))  # Print the scaled data for a preview

    # Create a DataFrame with scaled features and proper column names
    data_feat = pd.DataFrame(scaler_features, columns=data.columns[:-1])
    print(data_feat.head(15))  # Print the scaled features with column names for verification

    # Split the data into training and test sets (80% train, 20% test)
    # X contains the features, y contains the target class
    X_train, X_test, y_train, y_test = train_test_split(
        scaler_features, data["TARGET CLASS"], test_size=0.2, random_state=42  # Added random_state for reproducibility
    )

    # Create a KNeighborsClassifier with 5 neighbors (default K value)
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the classifier on the training data
    knn.fit(X_train, y_train)

    # Use the trained model to make predictions on the test set
    predictions = knn.predict(X_test)

    # Output the confusion matrix to evaluate the model's performance
    print(confusion_matrix(y_test, predictions))

    # Output a classification report for a detailed performance analysis
    print(classification_report(y_test, predictions))

    # Test different k values (from 1 to 41) to evaluate performance
    k_values = range(1, 41)  # Define range of k values to test
    error_rate = []  # List to store error rates for different k values

    for k in k_values:
        # Create a KNeighborsClassifier with the current k value
        knn = KNeighborsClassifier(n_neighbors=k)
        # Train the classifier using the training data
        knn.fit(X_train, y_train)
        # Make predictions on the test set
        pred_i = knn.predict(X_test)
        # Calculate and append the error rate for the current k value
        error_rate.append(np.mean(pred_i != y_test))

    # Plot the error rates against the tested k values
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, error_rate, color="blue", linestyle="dashed", marker="o", markerfacecolor="red",
             markersize=10)  # Fixed x-axis range to use `k_values`
    plt.title("Error Rate vs. K Value")  # Title of the plot
    plt.xlabel("K")  # Label for the x-axis
    plt.ylabel("Error Rate")  # Label for the y-axis
    plt.show()


if __name__ == "__main__":
    main()
