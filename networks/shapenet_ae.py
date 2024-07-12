import os
from torch import nn
import torch

from data.neural_field_datasets_shapenet import ImageTransform3D, ShapeNetDataset

class VanillaDecoder(nn.Module):
    def __init__(self,
                 dimensions = (64, 32, 16),
                 activation: nn.Module = nn.GELU()  # activation to be used
                 ):
        super().__init__()
        
        layers = []
        
        for i in range(len(dimensions) - 2):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            layers.append(activation)
            
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
       
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.decoder(x)

class VanillaEncoder(nn.Module):
    def __init__(self,
                 dimensions: list =  [64, 32, 16],
                 activation: nn.Module = nn.GELU()  # activation to be used
                 ):
        super().__init__()
        
        layers = []
        
        for i in range(len(dimensions) - 2):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            layers.append(activation)
            
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))

        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)
    
class PositionEncoder(nn.Module):
    def __init__(self,
                 num_emb: int,
                 emb_dim: int,
                 dimensions: list =  [64, 32, 16],
                 activation: nn.Module = nn.GELU()  # activation to be used
                 ):
        super().__init__()
        
        self.emb = nn.Embedding(num_emb, emb_dim)
        layers = []
        
        for i in range(len(dimensions) - 2):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            layers.append(activation)
            
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        if x.shape.__len__() == 3:
            embeddings = self.emb(x[:, :, -1].int())
            input = torch.cat((x[:, :, :-1], embeddings), dim=2)
            return torch.cat((self.encoder(input), x[:, :, -1:]), dim=2)
 
            
        else:
            embeddings = self.emb(x[:, -1].int())
            input = torch.cat((x[:, :-1], embeddings), dim=1)
            return torch.cat((self.encoder(input), x[:, -1:]), dim=1)
 
class LocalAutoencoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 num_layers_enc: int,
                 num_layers_dec: int, 
                 emb_dim: int, 
                 num_vec: int,
                 senquenz_length: int = 116,
                 ):
        super().__init__()
        reduction_factor_input = (input_dim - latent_dim) / num_layers_enc
        reduction_factor_emb = emb_dim/num_layers_enc
        dimensions_encoder = [input_dim + emb_dim]
        for i in range(num_layers_enc):
            dimensions_encoder.append(round(input_dim + emb_dim - (reduction_factor_input + reduction_factor_emb) *  (i + 1)))

        self.encoder = PositionEncoder(num_vec, emb_dim, dimensions_encoder)
        
        flattened_latent_dim = latent_dim * senquenz_length
        flattened_output_dim = input_dim * senquenz_length
        
        increasing_factor_latent = (flattened_output_dim - input_dim) / num_layers_dec

        dimensions_decoder = [flattened_latent_dim]
        for i in range(num_layers_dec - 1):
            dimensions_decoder.append(round(input_dim + emb_dim + (increasing_factor_latent - reduction_factor_emb) *  (i + 1)))
        
        dimensions_decoder.append(flattened_output_dim)

        self.decoder = VanillaDecoder(dimensions_decoder)
        
    def forward(self, x):
        latent = self.encoder(x)[:, :, :-1]
        latent_flattened = torch.flatten(latent, start_dim=1)
        latent_flattened = nn.functional.gelu(latent_flattened)
        return self.decoder(latent_flattened), latent_flattened
        

class GlobalAutoencoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim_local: int,
                 num_layers_enc_local: int,
                 num_layers_dec_global: int, 
                 emb_dim_local: int, 
                 num_vec_local: int,
                 senquenz_length: int = 116,
                 latent_dim_global: int = int(3712//2),
                 num_layer_enc_global = 2,
                 only_global_decode: bool = True,
                 
                 ):
        super().__init__()
        

        self.local_autoencoder = VanillaAutoencoder(input_dim, latent_dim_local, num_layers_enc_local, emb_dim_local, num_vec_local)

        
        flattened_latent_dim_local = latent_dim_local * senquenz_length

        
        if only_global_decode:
            self.flattened_encoder = nn.Identity(flattened_latent_dim_local, flattened_latent_dim_local)
            assert latent_dim_global == flattened_latent_dim_local, "Dimensions have to match!"
        else:
            # global latent projection
            reduction_factor_input = (flattened_latent_dim_local - latent_dim_global) / num_layer_enc_global
            dimensions_global_encoder = [flattened_latent_dim_local]
            for i in range(num_layer_enc_global - 1):
                dimensions_global_encoder.append(round(flattened_latent_dim_local - reduction_factor_input  *  (i + 1)))
            dimensions_global_encoder.append(latent_dim_global)
            
            self.flattened_encoder = VanillaEncoder(dimensions_global_encoder)
        
        flattened_latent_dim = latent_dim_global
        flattened_output_dim = input_dim * senquenz_length
        
        increasing_factor_latent = (flattened_output_dim - flattened_latent_dim) / num_layers_dec_global

        dimensions_decoder = [flattened_latent_dim]
        for i in range(num_layers_dec_global - 1):
            dimensions_decoder.append(round(flattened_latent_dim + (increasing_factor_latent) *  (i + 1)))
        
        dimensions_decoder.append(flattened_output_dim)

        self.decoder = VanillaDecoder(dimensions_decoder)

        
        
    def forward(self, x):
        reconstructed, latent  = self.local_autoencoder(x)
        latent = latent[:, :, :-1]
        latent_flattened = torch.flatten(latent, start_dim=1)
        latent_flattened = nn.functional.gelu(latent_flattened)
        latent_flattened = self.flattened_encoder(latent_flattened)
        latent_flattened = nn.functional.gelu(latent_flattened)
        return self.decoder(latent_flattened), reconstructed[:, :, :-1]
        
    
 
 
    
class VanillaAutoencoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 num_layers: int,
                 emb_dim: int, 
                 num_vec: int,
                 ):
        super().__init__()
        reduction_factor_input = (input_dim - latent_dim) / num_layers
        reduction_factor_emb = emb_dim/num_layers


        dimensions_encoder = [input_dim + emb_dim]
        for i in range(num_layers):
            dimensions_encoder.append(round(input_dim + emb_dim - (reduction_factor_input + reduction_factor_emb) *  (i + 1)))

        self.encoder = PositionEncoder(num_vec, emb_dim, dimensions_encoder)

        output_dim = input_dim
        input_dim = latent_dim


        increasing_factor_latent = (output_dim - input_dim) / num_layers

        dimensions_decoder = [input_dim + emb_dim]
        for i in range(num_layers):
            dimensions_decoder.append(round(input_dim + emb_dim + (increasing_factor_latent - reduction_factor_emb) *  (i + 1)))



        self.decoder = PositionEncoder(num_vec, emb_dim, dimensions_decoder)


        
    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent), latent
 

if __name__ == "__main__":
    encoder = VanillaEncoder(512, 5, 2, 2)
    x = torch.rand(16, 512)
    dataset_weights = ShapeNetDataset(os.path.join("./", "datasets", "shapenet_nefs", "pretrained"), transform=ImageTransform3D())
    print(dataset_weights[0].shape)
    


        
        
        
        