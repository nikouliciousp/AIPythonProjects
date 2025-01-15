import numpy as np

# 1. numpy.array(): Create an array from a list or tuple.
array_example = np.array([1, 2, 3])
print("Example of numpy.array():", array_example)

# 2. numpy.linspace(): Generate evenly spaced values.
linspace_example = np.linspace(0, 10, 5)
print("Example of numpy.linspace():", linspace_example)

# 3. numpy.zeros(): Create an array filled with zeros.
zeros_example = np.zeros((2, 3))
print("Example of numpy.zeros():", zeros_example)

# 3a. numpy.eye(): Create an identity matrix.
eye_example = np.eye(3)
print("Example of numpy.eye():", eye_example)

# 4. numpy.ones(): Create an array filled with ones.
ones_example = np.ones((3, 2))
print("Example of numpy.ones():", ones_example)

# 4a. numpy.full(): Create an array filled with a specific value.
full_example = np.full((2, 2), 7)
print("Example of numpy.full():", full_example)

# 5. numpy.reshape(): Change the shape of an array.
reshape_example = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
print("Example of numpy.reshape():", reshape_example)

# 5a. numpy.transpose(): Compute the transpose of a matrix.
transpose_example = np.transpose(np.array([[1, 2], [3, 4], [5, 6]]))
print("Example of numpy.transpose():", transpose_example)

# 5b. numpy.flatten(): Flatten a multi-dimensional array.
flatten_example = np.array([[1, 2], [3, 4]]).flatten()
print("Example of numpy.flatten():", flatten_example)

# 6. numpy.mean(): Compute the mean of the array.
mean_example = np.mean([1, 2, 3, 4, 5])
print("Example of numpy.mean():", mean_example)

# 6a. numpy.median(): Compute the median of the array.
median_example = np.median([1, 2, 3, 4, 5])
print("Example of numpy.median():", median_example)

# 7. numpy.sum(): Compute the sum of array elements.
sum_example = np.sum([1, 2, 3, 4, 5])
print("Example of numpy.sum():", sum_example)

# 7a. numpy.dot(): Compute the dot product of two arrays.
dot_example = np.dot([1, 2, 3], [4, 5, 6])
print("Example of numpy.dot():", dot_example)

# 7b. numpy.cross(): Compute the cross product of two vectors.
cross_example = np.cross([1, 0, 0], [0, 1, 0])
print("Example of numpy.cross():", cross_example)

# 8. numpy.max(): Compute the maximum value.
max_example = np.max([1, 2, 3, 4, 5])
print("Example of numpy.max():", max_example)

# 8a. numpy.argmax(): Find the index of the maximum value.
argmax_example = np.argmax([1, 2, 3, 4, 5])
print("Example of numpy.argmax():", argmax_example)

# 9. numpy.min(): Compute the minimum value.
min_example = np.min([1, 2, 3, 4, 5])
print("Example of numpy.min():", min_example)

# 9a. numpy.argmin(): Find the index of the minimum value.
argmin_example = np.argmin([1, 2, 3, 4, 5])
print("Example of numpy.argmin():", argmin_example)

# 9b. numpy.std(): Compute the standard deviation of the array.
std_example = np.std([1, 2, 3, 4, 5])
print("Example of numpy.std():", std_example)

# 10. numpy.random.randn(): Generate random numbers with a normal distribution.
random_example = np.random.randn(3, 2)
print("Example of numpy.random.randn():", random_example)

# 10a. numpy.random.randint(): Generate random integers within a range.
randint_example = np.random.randint(0, 10, (3, 3))
print("Example of numpy.random.randint():", randint_example)

# 10b. numpy.cumsum(): Compute the cumulative sum of array elements.
cumsum_example = np.cumsum([1, 2, 3, 4, 5])
print("Example of numpy.cumsum():", cumsum_example)

# 10c. numpy.cumprod(): Compute the cumulative product of array elements.
cumprod_example = np.cumprod([1, 2, 3, 4, 5])
print("Example of numpy.cumprod():", cumprod_example)

# 11. numpy slicing: Extract elements from an array.
slicing_example = np.array([10, 20, 30, 40])[1:3]  # Extract elements 20 and 30
print("Example of slicing:", slicing_example)

# 12. numpy.concatenate(): Concatenate arrays along an axis.
concatenate_example = np.concatenate((np.array([1, 2]), np.array([3, 4])), axis=0)
print("Example of numpy.concatenate():", concatenate_example)

# 13. numpy.stack(): Stack arrays along a new axis.
stack_example = np.stack((np.array([1, 2]), np.array([3, 4])), axis=0)
print("Example of numpy.stack():", stack_example)

# 13a. numpy.hstack(): Stack arrays horizontally (axis=1).
hstack_example = np.hstack((np.array([[1], [2]]), np.array([[3], [4]])))
print("Example of numpy.hstack():", hstack_example)

# 13b. numpy.vstack(): Stack arrays vertically (axis=0).
vstack_example = np.vstack((np.array([1, 2]), np.array([3, 4])))
print("Example of numpy.vstack():", vstack_example)

# 13c. numpy.dstack(): Stack arrays along the third axis.
dstack_example = np.dstack((np.array([[1, 2]]), np.array([[3, 4]])))
print("Example of numpy.dstack():", dstack_example)

# 14. numpy.split(): Split an array into multiple sub-arrays.
split_example = np.split(np.array([1, 2, 3, 4]), 2)
print("Example of numpy.split():", split_example)

# 15. numpy.where(): Return the indices of elements that satisfy a condition.
where_example = np.where(np.array([1, 2, 3, 4]) > 2)
print("Example of numpy.where():", where_example)

# 16. numpy.searchsorted(): Find the indices for insertion in a sorted array.
searchsorted_example = np.searchsorted([1, 3, 4, 7], 3)
print("Example of numpy.searchsorted():", searchsorted_example)

# 17. numpy.sort(): Sort the elements of an array.
sort_example = np.sort(np.array([3, 1, 4, 1, 5]))
print("Example of numpy.sort():", sort_example)

# 18. NumPy filtering with a boolean mask.
filter_example = np.array([1, 2, 3, 4])[np.array([True, False, True, False])]
print("Example of NumPy filtering:", filter_example)

# 19. numpy.random.randint(): Generate random integers within a range.
randint_example = np.random.randint(0, 10, (3, 3))
print("Example of numpy.random.randint():", randint_example)

# 20. numpy.random.rand(): Generate random numbers in a range [0, 1).
rand_example = np.random.rand(2, 2)
print("Example of numpy.random.rand():", rand_example)

# 21. numpy.random.randn(): Generate random numbers with a normal distribution.
randn_example = np.random.randn(2, 2)
print("Example of numpy.random.randn():", randn_example)

# 22. numpy.random.choice(): Randomly select from a given array.
choice_example = np.random.choice([10, 20, 30], size=2, replace=False)
print("Example of numpy.random.choice():", choice_example)

# 23. numpy.random.shuffle(): Shuffle an array in place.
shuffle_example = np.array([1, 2, 3, 4])
np.random.shuffle(shuffle_example)
print("Example of numpy.random.shuffle():", shuffle_example)

# 24. numpy.arange(): Generate values within a range.
arange_example = np.arange(0, 10, 2)
print("Example of numpy.arange():", arange_example)

# 25. Additional numpy.linspace(): Generate evenly spaced numbers for educational purpose.
linspace_additional_example = np.linspace(0, 5, 10)
print("Additional example of numpy.linspace():", linspace_additional_example)