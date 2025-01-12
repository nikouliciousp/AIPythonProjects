def main():
    # Prompt the user for input strings
    str1 = input("Enter the first string: ")
    str2 = input("Enter the second string: ")

    # Ensure the strings have the same length
    if len(str1) != len(str2):
        print("Error: Strings should be of the same length.")
        return

    # Calculate Hamming distance
    hamming_distance = sum(c1 != c2 for c1, c2 in zip(str1, str2))
    print(f"The Hamming distance is: {hamming_distance}")

if __name__ == "__main__":
    main()
