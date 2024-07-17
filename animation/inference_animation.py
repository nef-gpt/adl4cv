file = "./submissions/animation-factory/media/logits_last.npy"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import softmax


plt.style.use("dark_background")

# plt.patch.set_facecolor("#121212")


# Load logits from file
logits = np.load(file)

# Ensure logits is a 2D array for softmax function
logits = logits.reshape(1, -1)

top_k = 16


# Define the softmax function with temperature
def softmax_with_temperature(logits, temperature):
    return np.sort(softmax(temperature * logits, axis=1))[:, ::-1][:, :top_k]


# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
categories = np.arange(logits.shape[1])
fig.patch.set_facecolor("#121212")
ax.set_facecolor("#121212")
ax.set_ylim(0, 0.4)
ax.set_xlim(-1, top_k)

# we want to have the idxes of the top_k values in logits[1] after sorting them
xticklabels = np.argsort(logits[0])[::-1][:top_k]

ax.set_xticks(categories[:top_k])
ax.set_xticklabels(xticklabels)

bars = ax.bar(categories, np.zeros_like(categories))

# Define the range of temperatures
temperatures = np.concatenate(
    [np.linspace(1, 0.1, num=100), np.linspace(0.1, 2, num=200)]
)


# Set the face color of the figure and axes to transparent
# fig.patch.set_facecolor("#121212")
# ax.patch.set_facecolor("#121212")
# ax.patch.set_alpha(0.0)


def update(frame):
    temperature = temperatures[frame]
    transformed_logits = softmax_with_temperature(logits, temperature)
    for bar, new_value in zip(bars, transformed_logits[0]):
        bar.set_height(new_value)
    ax.set_title(f"Temperature: {temperature:.2f} / Top K: {top_k}")
    fig.patch.set_facecolor("#121212")  # Ensure background color is set for each frame
    return bars


# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(temperatures), blit=True)


# Save the animation as an MP4 file
FFwriter = animation.FFMpegWriter(fps=30)
ani.save("animated_bar_chart.mp4", writer=FFwriter)

# Save the animation as a GIF file
# ani.save("animated_bar_chart.gif", writer="imagemagick", fps=30)

plt.show()
