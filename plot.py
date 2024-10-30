import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot on the first subplot
ax1.plot(x, y1, label='Sine', color='blue', linestyle='-')
ax1.set_xlabel("X-axis (0 to 10)")          # Set X-axis label
ax1.set_ylabel("Y-axis (Sine values)")      # Set Y-axis label
ax1.set_xlim(0, 10)                         # Set X-axis limits
ax1.set_ylim(-1.5, 1.5)                     # Set Y-axis limits
ax1.set_title("Sine Wave")                  # Set title for the subplot
ax1.legend(loc="upper right")               # Add legend to the first plot

# Plot on the second subplot
ax2.plot(x, y2, label='Cosine', color='red', linestyle='--')
ax2.set_xlabel("X-axis (0 to 10)")          # Set X-axis label
ax2.set_ylabel("Y-axis (Cosine values)")    # Set Y-axis label
ax2.set_xlim(0, 10)                         # Set X-axis limits
ax2.set_ylim(-1.5, 1.5)                     # Set Y-axis limits
ax2.set_title("Cosine Wave")                # Set title for the subplot
ax2.legend(loc="upper right")               # Add legend to the second plot

# Adjust layout for better spacing
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig("sine_cosine_plot.png")

# Show the plot
plt.show()
