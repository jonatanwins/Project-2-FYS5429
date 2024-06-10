import json
import numpy as np
import os

# Create a dummy 2D array
dummy_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Create a dummy mask
mask = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

# Apply the mask to the array
masked_array = dummy_array * mask

# Convert the masked array to a list for JSON serialization
masked_array_list = masked_array.tolist()

# Define the directory and file name
log_dir = ""
file_name = "test.json"
file_path = os.path.join(log_dir, file_name)

# Dump the masked array to a JSON file with nice formatting
with open(file_path, 'w') as f:
    json.dump(masked_array_list, f, indent=4)

print(f"Masked array dumped to {file_path}")

# Print the masked array and the JSON content for verification
print("Masked Array:")
print(masked_array)

with open(file_path, 'r') as f:
    print("\nContent of the JSON file:")
    print(f.read())
