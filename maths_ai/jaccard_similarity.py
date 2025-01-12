def jaccard_similarity(set1, set2):
    """
    Calculate the Jaccard similarity between two sets.
    Jaccard similarity = (Intersection of sets) / (Union of sets)
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

def main():
    # Example usage
    set1 = {"apple", "banana", "cherry"}
    set2 = {"banana", "cherry", "date"}
    similarity = jaccard_similarity(set1, set2)
    print(f"Jaccard Similarity: {similarity:.2f}")

if __name__ == "__main__":
    main()