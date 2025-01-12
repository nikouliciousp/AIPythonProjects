import math


def calculate_euclidean_distance(dimensions, coord1, coord2):
    """
    Calculates the Euclidean distance between two points in 1D, 2D, or 3D space.

    :param dimensions: int, the dimension of the space (1, 2, or 3).
    :param coord1: list, the coordinates of the first point.
    :param coord2: list, the coordinates of the second point.
    :return: float, the Euclidean distance between the two points.
    """
    if dimensions == 1:
        return abs(coord2[0] - coord1[0])
    elif dimensions == 2:
        return math.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)
    elif dimensions == 3:
        return math.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2 + (coord2[2] - coord1[2]) ** 2)
    else:
        raise ValueError("Invalid dimensions. Only 1D, 2D, and 3D are supported.")


def main():
    """
    Main function to prompt the user for input, calculate the Euclidean distance,
    and display the result.
    """
    dimensions = int(input("Enter the dimension (1 for 1D, 2 for 2D, 3 for 3D): "))

    if dimensions == 1:
        x1 = float(input("Enter the coordinate x1: "))
        x2 = float(input("Enter the coordinate x2: "))
        coord1, coord2 = [x1], [x2]
    elif dimensions == 2:
        x1 = float(input("Enter the coordinate x1: "))
        y1 = float(input("Enter the coordinate y1: "))
        x2 = float(input("Enter the coordinate x2: "))
        y2 = float(input("Enter the coordinate y2: "))
        coord1, coord2 = [x1, y1], [x2, y2]
    elif dimensions == 3:
        x1 = float(input("Enter the coordinate x1: "))
        y1 = float(input("Enter the coordinate y1: "))
        z1 = float(input("Enter the coordinate z1: "))
        x2 = float(input("Enter the coordinate x2: "))
        y2 = float(input("Enter the coordinate y2: "))
        z2 = float(input("Enter the coordinate z2: "))
        coord1, coord2 = [x1, y1, z1], [x2, y2, z2]
    else:
        raise ValueError("Invalid dimension selected. Please pick 1, 2, or 3.")

    distance = calculate_euclidean_distance(dimensions, coord1, coord2)
    print(f"The Euclidean distance is: {distance}")


if __name__ == "__main__":
    main()
