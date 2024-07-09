import os

from data.pointcloud_dataset import PointCloud

files = [file for file in os.listdir("./datasets/02691156_mesh")]

for file in files:
    PointCloud(file, 1)
    break