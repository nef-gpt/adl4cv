import os
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the directory to save the images
save_dir = "./submissions/presentation_1/public/mnist_gt"

# Create the directory if it does not exist
os.makedirs(save_dir, exist_ok=True)

# Define the transformation to convert the images to tensors
transform = transforms.Compose([transforms.ToTensor()])

# Download the MNIST dataset
mnist_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

# Indices of the samples to retrieve
sample_indices = [0, 11, 35, 47, 65]

# Loop over the indices and save the images
for index in sample_indices:
    image, label = mnist_dataset[index]
    image_path = os.path.join(save_dir, f"mnist_{index}.png")
    # Convert the tensor to a numpy array and save the image
    plt.imsave(image_path, image.squeeze().numpy(), cmap="gray")

print("Images have been saved to:", save_dir)
