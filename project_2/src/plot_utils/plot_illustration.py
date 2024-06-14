import matplotlib.pyplot as plt
import numpy as np

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Hide axes
ax.axis("off")

# Define colors
colors = ["#F9D5A7", "#F19C79", "#CE4B27"]
gray = "#B0B0B0"
dot_positions_theta = [
    (0, 2.5),
    (0.5, 2.0),
    (1, 1.5),
    (1.5, 1.0),
    (2, 0.5),
    (2.5, 0.0),
    (3, -0.5),
]
dot_positions_xi = [
    (0, 2.5),
    (0.5, 2.0),
    (1, 1.5),
    (1.5, 1.0),
    (2, 0.5),
    (2.5, 0.0),
    (3, -0.5),
]

# Draw the Z vector
for i, color in enumerate(colors):
    ax.plot([0, 0], [2.5 - i, 3.5 - i], color=color, linewidth=20)
ax.text(-0.2, 2.5, r"$\dot{Z}$", fontsize=18, va="center")

# Draw the Theta(Z) matrix
for i in range(7):
    ax.plot([1, 1], [2.5 - i * 0.5, 3 - i * 0.5], color=gray, linewidth=20)
    if i < len(dot_positions_theta):
        ax.plot(
            [1],
            [dot_positions_theta[i][1]],
            "o",
            color=colors[i % len(colors)],
            markersize=10,
            alpha=0.7,
        )
ax.text(1.2, 2.5, r"$\Theta(Z)$", fontsize=18, va="center")
ax.text(1, 1.0, "...", fontsize=18, ha="center")

# Draw the Xi matrix
for i in range(3):
    for j in range(3):
        ax.plot(
            [2, 2],
            [2.5 - j * 0.5, 3 - j * 0.5],
            color=colors[j],
            linewidth=20,
            alpha=0.7,
        )
        if i == j:
            ax.plot(2, 2.75 - j * 0.5, "o", color=colors[j], markersize=10)
ax.text(2.2, 2.5, r"$\Xi$", fontsize=18, va="center")

# Annotations for detailed elements
ax.text(2.3, 3.25, r"$\xi_{1}$", fontsize=12, va="center")
ax.text(2.3, 2.75, r"$\xi_{2}$", fontsize=12, va="center")
ax.text(2.3, 2.25, r"$\xi_{3}$", fontsize=12, va="center")

# Set limits and aspect
ax.set_xlim(-0.5, 2.8)
ax.set_ylim(-1, 4)
ax.set_aspect("equal")

# Display the plot
plt.show()
