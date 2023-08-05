import numpy as np
import matplotlib.pyplot as plt

# Define the range of p values
e_values = [0.15,0.2, 0.25, 0.35, 0.45,0.5, 0.707, 1, 2.34, 4, 8,15 ,28,35, 800]

# Define the grid of x and y coordinates
x = np.linspace(-1.5, 1.5, 1000)
y = np.linspace(-1.5, 1.5, 1000)
X, Y = np.meshgrid(x, y)

# Define the number of rows and columns for subplots
num_rows = 3
num_cols = 5

# Create a figure with subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Generate a separate graph for each value of p and plot it in the corresponding subplot
for i, e in enumerate(e_values):
    # Calculate the Minkowski distance for each point from the origin in the grid for the current p value
    distance = np.power(np.power(np.abs(X), e) + np.power(np.abs(Y), e), 1 / e)
    
    # Determine the row and column index for the current subplot
    row_idx = i // num_cols
    col_idx = i % num_cols
    
    # Plot the graph in the corresponding subplot
    axs[row_idx, col_idx].contour(X, Y, distance, levels=[1.4], colors='b')
    axs[row_idx, col_idx].axhline(0, color='black', linewidth=0.5)
    axs[row_idx, col_idx].axvline(0, color='black', linewidth=0.5)
    axs[row_idx, col_idx].set_xlim(-2, 2)
    axs[row_idx, col_idx].set_ylim(-2, 2)
    axs[row_idx, col_idx].set_aspect('equal')
    axs[row_idx, col_idx].set_title(f"Minkowski Distance (e = {e})")
    axs[row_idx, col_idx].set_xlabel("x")
    axs[row_idx, col_idx].set_ylabel("y")

# Adjust the spacing between subplots
fig.tight_layout()
fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.1)

# Display the figure
plt.show()
