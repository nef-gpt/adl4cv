import glob
import os
import numpy as np
import torch
from tqdm import tqdm
import trimesh

from utils import get_default_device
from utils.hd_utils import calculate_fid_3d
from utils.metrics import compute_all_metrics, compute_all_metrics_3D

import pandas as pd


def calc_metrics(split_type):
    num_points = 2048
    dataset_path = os.path.join(
        "datasets",
        "02691156" + f"_{num_points}_pc",
    )
    test_object_names = np.genfromtxt(
        os.path.join(dataset_path, f"{split_type}_split.lst"), dtype="str"
    )
    print("test_object_names.length", len(test_object_names))

    # orig_meshes_dir = f"orig_meshes/run_{wandb.run.name}"
    # os.makedirs(orig_meshes_dir, exist_ok=True)

    # During validation, only use some of the val and train shapes for speed
    num_samples = 32
    num_epochs = 32

    # if split_type == "val" and num_samples is not None:
    #     test_object_names = test_object_names[:num_samples]
    # elif split_type == "train" and num_samples is not None:
    #     test_object_names = test_object_names[:num_samples]
    n_points = num_points

    # First process ground truth shapes
    pcs = []
    for obj_name in test_object_names:
        pc = np.load(os.path.join(dataset_path, obj_name + ".npy"))
        pc = pc[:, :3]

        pc = torch.tensor(pc).float()
        # TODO: do we need to do this as well (do a 0.5 normalization?)
        pc = pc.float()
        shift = pc.mean(dim=0).reshape(1, 3)
        scale = pc.flatten().std().reshape(1, 1)
        pc = (pc - shift) / scale
        pcs.append(pc)
    # r = Rotation.from_euler("x", 90, degrees=True)
    # self.logger.experiment.log({"3d_gt": wandb.Object3D(r.apply(np.array(pcs[0])))})
    ref_pcs = torch.stack(pcs)

    # We are generating slightly more than ref_pcs
    # number_of_samples_to_generate = int(len(ref_pcs) * self.cfg.test_sample_mult)

    grid = [
        # (1, 1),
        (3, 0.8),
        (3, 1),
        (5, 0.8),
        (5, 1),
    ]

    device = get_default_device()

    print("Running configurations")
    # {'lgan_mmd-CD': tensor(0.0092, device='cuda:0'), 'lgan_cov-CD': tensor(0.4375, device='cuda:0'), 'lgan_mmd_smp-CD': tensor(0.0093, device='cuda:0'), '1-NN-CD-acc_t': tensor(0.9062, device='cuda:0'), '1-NN-CD-acc_f': tensor(0.7500, device='cuda:0'), '1-NN-CD-acc': tensor(0.8281, device='cuda:0'), 'fid': 22.617727279663086}
    # we want to store the grid plus all of the metrics in a pandas dataframe
    results = pd.DataFrame(columns=["top_k", "temperature", "epoch", "lgan_mmd-CD", "lgan_cov-CD", "lgan_mmd_smp-CD", "1-NN-CD-acc_t", "1-NN-CD-acc_f", "1-NN-CD-acc", "fid"])
    for top_k, temperate in tqdm(grid):
        # all files matching glob
        files = glob.glob(
            f"./models/meshes/{split_type}_temperature_{temperate}_top_{top_k}*.obj"
        )
        assert len(files) >= 0
        sample_batch = []
        for file in files:
            mesh = trimesh.load_mesh(file)
            # vert = mesh.vertices
            # vert = vert - np.mean(vert, axis=0, keepdims=True)
            # v_max = np.amax(vert)
            # v_min = np.amin(vert)
            # vert *= 0.95 / (max(abs(v_min), abs(v_max)))
            # mesh.vertices = vert

            pc = torch.tensor(mesh.sample(n_points))
            pc = pc.float()

            pc = pc * 2

            pc = pc.float()
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
            pc = (pc - shift) / scale

            sample_batch.append(pc)
        
        sample_pcs = torch.stack(sample_batch)

        for epoch in tqdm(range(num_epochs)):
            # randomly sample from sample_pcs and ref_pcs
            # num_samples times
            epoch_sample_pcs = sample_pcs[torch.randint(0, len(sample_pcs), (num_samples,))]
            epoch_ref_pcs = ref_pcs[torch.randint(0, len(ref_pcs), (num_samples,))]
            
            assert epoch_sample_pcs.size(0) == epoch_ref_pcs.size(0)
            print("Starting metric computation for", split_type, temperate, top_k)

            fid = calculate_fid_3d(epoch_sample_pcs.to(device), epoch_ref_pcs.to(device))
            metrics = compute_all_metrics_3D(
                epoch_sample_pcs.to(device),
                epoch_ref_pcs.to(device),
                16 if split_type == "test" else 16,
            )

            # unwrap tensors in metrics
            for key in metrics.keys():
                metrics[key] = metrics[key].item()

            metrics["fid"] = fid.item()

            print("Completed metric computation for", split_type, temperate, top_k)

            print(metrics)

            # store the results in the dataframe
            
            results = pd.concat([results, pd.DataFrame([{"top_k": top_k, "temperature": temperate, "epoch": epoch, **metrics}])])

    results.to_csv(f"./models/{split_type}_metrics_z_score.csv")

if __name__ == "__main__":
    calc_metrics("train")
