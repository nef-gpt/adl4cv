import torch
import itertools
import copy
import numpy as np
from networks.mlp_models import MLP3D

def permute_weights_together(weights1, weights2, permutation):
    """
    Permutes the weights of two layers together according to the given permutation.
    """
    P = torch.eye(weights1.shape[0])[permutation]
    permuted_weights1 = torch.mm(P.T, weights1)
    permuted_weights2 = torch.mm(weights2, P)
    return permuted_weights1, permuted_weights2

def apply_permutation_to_model(state_dict, layer1_name, layer2_name, permutation):
    """
    Applies the permutation to the weights and biases of the specified layers.
    """
    permuted_state_dict = copy.deepcopy(state_dict)
    
    # Permute the weights
    weight1_key = layer1_name + ".weight"
    weight2_key = layer2_name + ".weight"
    bias1_key = layer1_name + ".bias"
    bias2_key = layer2_name + ".bias"
    
    permuted_weights1, permuted_weights2 = permute_weights_together(
        state_dict[weight1_key], state_dict[weight2_key], permutation)
    
    permuted_state_dict[weight1_key] = permuted_weights1
    permuted_state_dict[weight2_key] = permuted_weights2
    
    # Permute the biases (only if it's not the input layer)
    if bias1_key in state_dict:
        permuted_state_dict[bias1_key] = state_dict[bias1_key][permutation]
    if bias2_key in state_dict:
        permuted_state_dict[bias2_key] = state_dict[bias2_key][permutation]
    
    return permuted_state_dict

def generate_permuted_models(model_path, layer1_name, layer2_name, num_permutations):
    """
    Loads the model from the specified path, applies permutations to the weights
    of the specified layers, and saves the permuted models.
    """
    # Load the model state dictionary
    state_dict = torch.load(model_path)
    
    # Get the number of neurons in the specified layer
    num_neurons = state_dict[layer1_name + ".weight"].shape[1]
    
    # Generate permutations
    permutations = list(itertools.permutations(range(num_neurons)))
    
    # Limit to the specified number of permutations
    if num_permutations < len(permutations):
        permutations = permutations[:num_permutations]
    
    # Create and save permuted models
    permuted_models = []
    for i, permutation in enumerate(permutations):
        permuted_state_dict = apply_permutation_to_model(state_dict, layer1_name, layer2_name, permutation)
        permuted_model_path = model_path.replace(".pth", f"_permuted_{i}.pth")
        torch.save(permuted_state_dict, permuted_model_path)
        permuted_models.append(permuted_model_path)
    
    return permuted_models

def main():
    model_path = "mnist-nerfs/unstructured/mnist-nerfs-unstructured-0_model_final.pth"  # Replace with your model path
    layer1_name = "layers.0.weight"  # Replace with the first layer name
    layer2_name = "layers.1.weight"  # Replace with the second layer name
    num_permutations = 5  # Number of permutations to generate and save
    
    permuted_models = generate_permuted_models(model_path, layer1_name, layer2_name, num_permutations)
    
    print(f"Generated permuted models: {permuted_models}")

if __name__ == "__main__":
    main()
