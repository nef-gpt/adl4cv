import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# Generate random data for demonstration
image = np.random.rand(28, 28)
W1 = np.random.rand(16, 18)
W2 = np.random.rand(16, 16)
W3 = np.random.rand(1, 16)
b1 = np.random.rand(16, 1)
b2 = np.random.rand(16, 1)
b3 = np.random.rand(1, 1)

# Log-loss curve data
x = np.linspace(0, 8, 400)
y = np.log(1 + np.exp(-x))

# Set the x-size of the whole figure
figure_x_size = 10
pixel_size = figure_x_size / 28

# Create the figure and axes using gridspec for better layout control
fig = plt.figure(figsize=(figure_x_size, figure_x_size * 3))
gs = gridspec.GridSpec(6, 3, height_ratios=[28, 16, 16, 1, 1, 28], width_ratios=[18, 1, 1])
gs.update(wspace=0.1, hspace=0.3)

# First row: Image
ax_image = plt.subplot(gs[0, 0])
ax_image.imshow(image, cmap='gray', aspect='equal')
ax_image.axis('off')

# Second row: W1 and b1
ax_W1 = plt.subplot(gs[1, 0])
im1 = ax_W1.imshow(W1, cmap='viridis', aspect='equal', vmin=0, vmax=1)
ax_W1.axis('off')

ax_b1 = plt.subplot(gs[1, 1])
ax_b1.imshow(b1, cmap='viridis', aspect='equal', vmin=0, vmax=1)
ax_b1.axis('off')

# Third row: W2 and b2
ax_W2 = plt.subplot(gs[2, 0])
ax_W2.imshow(W2, cmap='viridis', aspect='equal', vmin=0, vmax=1)
ax_W2.axis('off')

ax_b2 = plt.subplot(gs[2, 1])
ax_b2.imshow(b2, cmap='viridis', aspect='equal', vmin=0, vmax=1)
ax_b2.axis('off')

# Fourth row: W3 and b3
ax_W3 = plt.subplot(gs[3, 0])
ax_W3.imshow(W3, cmap='viridis', aspect='equal', vmin=0, vmax=1)
ax_W3.axis('off')

ax_b3 = plt.subplot(gs[3, 1])
ax_b3.imshow(b3, cmap='viridis', aspect='equal', vmin=0, vmax=1)
ax_b3.axis('off')

# Fifth row: Colorbar
"""ax_cbar = plt.subplot(gs[4, 0])
cbar = fig.colorbar(im1, cax=ax_cbar, orientation='horizontal')
cbar.set_label('Intensity')
"""
# Sixth row: Log-loss curve (same size as the first row image)
"""ax_loss = plt.subplot(gs[5, 0])
ax_loss.plot(x, y, label='Log-Loss Curve', color='magenta')
ax_loss.legend()
ax_loss.set_title('Log-Loss Curve', fontsize=12)
ax_loss.set_xlim(0, 8)
ax_loss.set_aspect(aspect='auto')"""

# Save the figure
plt.rcParams.update(
    {
        "figure.facecolor": (0.0, 0.0, 0.0, 0.0),
        "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
    }
)
plt.savefig('visualization.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
