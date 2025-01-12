import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    """
    The main function to load, analyze, and visualize the USA housing dataset.
    """
    # Load the dataset
    usa_housing = pd.read_csv("USA_Housing.csv")

    # Print the first 10 rows of the dataset
    print(usa_housing.head(10))

    # Display dataset information
    print(usa_housing.info())

    # Display dataset statistical summary
    print(usa_housing.describe())

    # Visualize the pairwise relationships in the dataset
    sns.pairplot(usa_housing)
    plt.show()


if __name__ == "__main__":
    main()
