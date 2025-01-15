import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def main():
    """
    The main function to load, analyze, and visualize the USA housing dataset.
    """
    # Load the dataset from a CSV file
    usa_housing = pd.read_csv("USA_Housing.csv")  # Replace with the path to your dataset

    # Print the first 10 rows of the dataset to get an overview of the data
    print(usa_housing.head(10))  # Useful for familiarizing with the data structure

    # Display dataset information, including data types and non-null values
    print(usa_housing.info())  # Identifies data types, missing values, etc.

    # Display basic statistical summary of the dataset (mean, median, etc.)
    print(usa_housing.describe())  # Basic statistical insights

    # Visualize the pairwise relationships in the dataset using scatterplots for quick analysis
    sns.pairplot(usa_housing)  # Pairwise scatterplots for correlations and trends
    plt.show()  # Display the pairplot

    # Visualize the distribution of the "Price" column with a kernel density estimate
    sns.displot(usa_housing["Price"], kde=True)  # Price distribution with density plot
    plt.show()  # Display the distribution

    # Print the column names of the dataset to list the available features
    print(usa_housing.columns)  # Lists all columns in the dataset

    # Define the input features (independent variables) used for prediction
    x = usa_housing[
        ["Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms",
         "Area Population"]]  # Feature selection for the model

    # Define the target variable (dependent variable) we want to predict
    y = usa_housing["Price"]  # Target variable (Price)

    # Split the dataset into training and testing sets (80% train, 20% test for evaluation)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)  # Train-test split

    # Initialize the Linear Regression model used for prediction
    linear_reg = LinearRegression()  # Instantiates the Linear Regression model

    # Train the Linear Regression model using the training data
    linear_reg.fit(x_train, y_train)  # Fits the model with training data

    # Create a DataFrame to display the model's coefficients corresponding to each feature
    coef_df = pd.DataFrame(linear_reg.coef_, x.columns, columns=["Coefficient"])  # Coefficients for each feature

    # Print the coefficients for interpretation (impact of each feature on the target variable)
    print(f"Coefficients: \n{coef_df}")  # Display the coefficients

    # Make predictions using the testing data
    predictions = linear_reg.predict(x_test)  # Generate predicted values for test set

    # Plot the actual prices versus the predicted prices for visual comparison
    plt.scatter(y_test, predictions, color="blue", alpha=0.5)  # Scatterplot of actual vs predicted values
    plt.plot(y_test, y_test, color="red")  # Reference line (y=x) for perfect prediction
    plt.xlabel("Actual Price")  # Label for the x-axis
    plt.ylabel("Predicted Price")  # Label for the y-axis
    plt.legend(["Actual Price", "Predicted Price"])  # Add a legend
    plt.title("Actual vs. Predicted Price")  # Title of the plot
    plt.show()  # Display the scatterplot

    # Calculate and print evaluation metrics
    print(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, predictions)}")  # Average magnitude of errors
    print(f"Mean Squared Error: {metrics.mean_squared_error(y_test, predictions)}")  # Mean squared error
    print(f"Root Mean Squared Error: {metrics.mean_squared_error(y_test, predictions)}")  # RMSE
    print(f"R2 Score: {r2_score(y_test, predictions):.2f}")  # R-squared statistic, a goodness-of-fit measure


if __name__ == "__main__":
    main()
