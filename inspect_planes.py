import torch


dict_planes = torch.load("./datasets/plane_mlp_weights/occ_1a04e3eab45ca15dd86060f189eb133_jitter_0_model_final.pth", map_location=torch.device('cpu'))

for key in dict_planes.keys():
    print(f"Dim of {key}: {dict_planes[key].size()}")

model_config = {
    "out_size": 0,
    
}