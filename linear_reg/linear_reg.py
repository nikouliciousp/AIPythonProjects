import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    """
    The main function to load, analyze, and visualize the USA housing dataset.
    """
    # Load the dataset from a CSV file
    usa_housing = pd.read_csv("USA_Housing.csv")

    # Print the first 10 rows of the dataset to get an overview
    print(usa_housing.head(10))

    # Display dataset information, including data types and non-null values
    print(usa_housing.info())

    # Display basic statistical summary of the dataset (mean, median, etc.)
    print(usa_housing.describe())

    # Uncomment the following lines to visualize the dataset
    # Visualize the pairwise relationships in the dataset (e.g., scatter plots)
    sns.pairplot(usa_housing)
    plt.show()

    # Visualize the distribution of the "Price" column with a kernel density estimate
    sns.displot(usa_housing["Price"], kde=True)
    plt.show()

    # Print the column names of the dataset to list available features
    print(usa_housing.columns)

    # Define the input features (independent variables) necessary for prediction
    x = usa_housing[
        ["Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms",
         "Area Population"]]

    # Define the target variable (dependent variable) we want to predict
    y = usa_housing["Price"]

    # Split the dataset into training and testing sets (80% train, 20% test for evaluation)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

    # Initialize the Linear Regression model
    linear_reg = LinearRegression()

    # Train the Linear Regression model with the training data
    linear_reg.fit(x_train, y_train)

    # Create a DataFrame to display the model's coefficients corresponding to each feature
    coef_df = pd.DataFrame(linear_reg.coef_, x.columns, columns=["Coefficient"])

    # Print the coefficients for interpretation of the model
    print(f"Coefficients: \n{coef_df}")


if __name__ == "__main__":
    main()
