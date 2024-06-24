import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data into a DataFrame
data = pd.read_csv('wandb_summery.csv')

plt.style.use('dark_background')

# Create a 2x2 plot with transparent background
fig, axs = plt.subplots(2, 2, figsize=(15, 10), facecolor='none')


# Set the background of the figure to be transparent
fig.patch.set_alpha(0.0)

# Plot MNIST Accuracy
axs[0, 0].plot(data['iter'][data['mnist_acc'].notna()], data['mnist_acc'][data['mnist_acc'].notna()], label='MNIST Accuracy')
axs[0, 0].set_title('MNIST Accuracy')
axs[0, 0].set_xlabel('Iterations')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].legend()

# Plot MNIST Cross-Entropy Loss
axs[0, 1].plot(data['iter'][data['mnist_acc'].notna()], data['mnist_loss'][data['mnist_acc'].notna()], label='MNIST Cross-Entropy Loss')
axs[0, 1].set_title('MNIST Cross-Entropy Loss')
axs[0, 1].set_xlabel('Iterations')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()

# Plot Training and Validation Loss
axs[1, 0].plot(data['iter'][data['train/loss'].notna()], data['train/loss'][data['train/loss'].notna()], label='Training Loss')
axs[1, 0].plot(data['iter'][data['train/loss'].notna()], data['val/loss'][data['train/loss'].notna()], label='Validation Loss')
axs[1, 0].set_title('Training and Validation Loss')
axs[1, 0].set_xlabel('Iterations')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].legend()

# Plot Validation Loss and Learning Rate
ax2 = axs[1, 1].twinx()
axs[1, 1].plot(data['iter'][data['lr'].notna()], data['val/loss'][data['val/loss'].notna()], label='Validation Loss')
axs[1, 1].set_title('Validation Loss and Learning Rate')
axs[1, 1].set_xlabel('Iterations')
axs[1, 1].set_ylabel('Validation Loss')
ax2.plot(data['iter'], data['lr'], label='Learning Rate')
ax2.set_ylabel('Learning Rate')
axs[1, 1].legend(loc='upper left')
ax2.legend(loc='upper right')

# Set the face color of the figure and axes to transparent
fig.patch.set_facecolor('none')
for ax in axs.flat:
    ax.patch.set_facecolor('none')
    ax.patch.set_alpha(0.0)

plt.tight_layout()
plt.show()
