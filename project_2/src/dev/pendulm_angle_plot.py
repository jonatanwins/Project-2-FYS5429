import matplotlib.pyplot as plt
import numpy as np


L = 1.0  
z = np.pi / 4  


x = L * np.sin(z)
y = -L * np.cos(z)


fig, ax = plt.subplots(figsize=(8, 8))


ax.plot([0, 0], [0, -L], 'k--', label='Vertical')

#
ax.plot([0, x], [0, y], 'b-', linewidth=2, label='Pendulum')


arc = np.linspace(0, z, 100)
ax.plot(0.1 * L * np.sin(arc), -0.1 * L * np.cos(arc), 'r', label='z')

# Calculate new position for the text annotation
text_x = 0.1 * L * np.sin(z / 2)
text_y = -0.1 * L * np.cos(z / 2) - 0.1  


ax.text(text_x, text_y, r'$z$', fontsize=12, ha='left', va='center')


ax.plot(x, y, 'ro', markersize=10)


ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')


ax.legend()
ax.axis('off')

# Show the plot
plt.show()
