import numpy as np

# Define a small array
small_array = np.array([[1, 2], [3, 4]])

# Repeat this array across each dimension
repeated_array = np.tile(small_array, (3, 4))  # Repeat 3 times along the first axis, 4 times along the second axis

print("Repeated Array:\n", repeated_array)
# Create an array of random elements
arr = np.array([15, 10, 20, 5])

# Get the indices that would sort the array
sorted_indices = np.argsort(arr)

print("Original Array:", arr)
print("Indices to Sort the Array in Ascending Order:", sorted_indices)
print("Sorted Array:", arr[sorted_indices])
