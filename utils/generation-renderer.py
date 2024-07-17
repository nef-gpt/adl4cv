from utils.hd_utils import render_image
from utils.visualization3d import model_to_mesh
from data.nef_shapenet_dataset import (
    ModelTransform3D,
    ShapeNetDataset,
    TokenTransform3D,
)
from vector_quantize_pytorch import ResidualVQ
from tqdm import tqdm


def main():
    # Directory structure is as follows
    # f"./models/generation/{by}_temperature_{temperature}_top_{top_k}/generation-{i}.pt"
    # where by is "train" or "val", temperature is a float, top_k and i an integer

    # glob pattern to load all files
    import glob
    import os
    import torch
    import re

    save_meshes = True

    # Output
    rq_dict = torch.load(
        "./models/rq_search_results/shapenet_retrained_learnable_rq_model_dim_1_vocab_127_batch_size_16_threshold_ema_dead_code_0_kmean_iters_1_num_quantizers_1_use_init_False.pth",
        map_location=torch.device("cpu"),
    )
    rq_dict.keys()

    state_dict = rq_dict["state_dict"]
    rq_config = rq_dict["rq_config"]

    rvq = ResidualVQ(**rq_config)
    rvq.load_state_dict(state_dict)

    token_transform = TokenTransform3D(rvq)
    dataset = ShapeNetDataset(
        os.path.join("./", "datasets", "shapenet_nef_2", "pretrained"), cpu=True
    )
    token_transform(*dataset[0])

    model_transform = ModelTransform3D(dataset[0][0]["model_config"])

    # Create output folder
    os.makedirs("./models/rendering", exist_ok=True)
    os.makedirs("./models/meshes", exist_ok=True)

    # Load all files
    files = glob.glob("./models/generation/*/*.pt")

    for file in tqdm(files, "Rendering"):
        # Load the file
        generated_tokens = torch.load(file, map_location=torch.device("cpu"))

        # get all the parameters from the file name
        # using a regex
        by, temperature, top_k, i = re.match(
            r".*/generation/(train|val)_temperature_([\d.]+)_top_(\d+)/generation-(\d+).pt",
            file,
        ).groups()

        if save_meshes:
            model_dict = token_transform.inverse(generated_tokens.unsqueeze(-1))
            model_quantized = model_transform(model_dict)[0]
            # model = dataset_model[index][0]
            mesh, sdf = model_to_mesh(model_quantized, res=256)
            mesh.export(
                f"./models/meshes/{by}_temperature_{temperature}_top_{top_k}_generation_{i}.obj"
            )
            continue

        output_file = f"./models/rendering/{by}_temperature_{temperature}_top_{top_k}_generation_{i}.png"

        if os.path.exists(output_file):
            continue

        model_dict = token_transform.inverse(generated_tokens.unsqueeze(-1))
        model_quantized = model_transform(model_dict)[0]
        # model = dataset_model[index][0]
        mesh, sdf = model_to_mesh(model_quantized, res=256)

        # Render the object
        render_image(mesh, path=output_file)


if __name__ == "__main__":
    main()
