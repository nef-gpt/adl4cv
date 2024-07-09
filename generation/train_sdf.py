import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the point cloud data from the .npy file
point_cloud = np.load('datasets/shapenet_pointcloud/fff513f407e00e85a9ced22d91ad7027.obj.npy')

print(point_cloud.shape)  # Check the shape of the loaded data

# Create a new figure for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Unpack the point cloud data
x = point_cloud[:, 0]
y = point_cloud[:, 1]
z = point_cloud[:, 2]
values = point_cloud[:, 3]

# Create masks for each class
mask_smaller = values < 0
mask_equal = values == 0
mask_bigger = values > 0

# Scatter plot of the point cloud for each class
ax.scatter(x[mask_smaller], y[mask_smaller], z[mask_smaller], c='r', s=1, label='Smaller than 0')  # Red for smaller than 0
ax.scatter(x[mask_equal], y[mask_equal], z[mask_equal], c='g', s=1, label='Equal to 0')            # Green for equal to 0
ax.scatter(x[mask_bigger], y[mask_bigger], z[mask_bigger], c='b', s=1, label='Bigger than 0')       # Blue for bigger than 0

# Set axis limits to be between -1 and 1
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add a legend
ax.legend()

# Show the plot
plt.show()
