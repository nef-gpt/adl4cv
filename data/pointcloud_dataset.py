from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm
import trimesh
import igl
import numpy as np
import os
import torch
from PIL import Image
import io

from utils.hd_utils import render_mesh


class PointCloud(Dataset):
    def __init__(
        self,
        path,
        on_surface_points,
        is_mesh=True,
        output_type="occ",
        out_act="sigmoid",
        n_points=100000,
        strategy="save_pc",
    ):
        super().__init__()
        self.output_type = output_type
        self.out_act = out_act
        print("Loading point cloud for ", path)
        self.vertices = None
        self.faces = None
        if is_mesh:
            if strategy == "save_pc":
                obj: trimesh.Trimesh = trimesh.load(path, process=False)

                # obj.show()

                vertices = obj.vertices

                vertices -= np.mean(vertices, axis=0, keepdims=True)
                v_max = np.amax(vertices)
                v_min = np.amin(vertices)
                vertices *= 0.5 * 0.95 / (max(abs(v_min), abs(v_max)))
                obj.vertices = vertices
                self.obj = obj

                color, depth = render_mesh(obj, res=(3600, 3600))

                fig = plt.figure()
                ax = fig.add_subplot()
                ax.set_xlim((500, 3000))
                ax.set_ylim((4250, 1250))
                ax.grid(False)
                ax.axis("off")
                ax.imshow(color)

                plt.show()

                n_points_uniform = n_points
                n_points_surface = n_points

                points_uniform = np.random.uniform(
                    -0.5, 0.5, size=(n_points_uniform, 3)
                )
                points_surface = obj.sample(n_points_surface)
                points_surface += 0.01 * np.random.randn(n_points_surface, 3)
                points = np.concatenate([points_surface, points_uniform], axis=0)

                trimesh.repair.fix_winding(obj)
                trimesh.repair.fix_inversion(obj)
                trimesh.repair.fill_holes(obj)

                # obj.show()
                inside_surface_values = igl.signed_distance(
                    points, obj.vertices, obj.faces
                )[0]

                # for debugging
                self.points = points
                self.inside_surface_values = inside_surface_values

                thresh = 0.005
                occupancies_winding = np.piecewise(
                    inside_surface_values,
                    [inside_surface_values <= thresh, inside_surface_values > thresh],
                    [-1.0, 1.0],
                )

                occupancies = occupancies_winding[..., None]
                print(points.shape, occupancies.shape, occupancies.sum())
                point_cloud = points
                point_cloud = np.hstack((point_cloud, occupancies))
                print(point_cloud.shape, points.shape, occupancies.shape)

        else:
            point_cloud = np.genfromtxt(path)
        print("Finished loading point cloud")

        pc_folder = "./datasets/02691156_pc"

        if strategy == "save_pc":
            self.coords = point_cloud[:, :3]
            self.occupancies = point_cloud[:, 3:]

            point_cloud_xyz = np.hstack((self.coords, self.occupancies))
            os.makedirs(pc_folder, exist_ok=True)
            np.save(
                pc_folder + "/" + path.split("/")[-1].split(".")[0], point_cloud_xyz
            )
        else:
            point_cloud = np.load(path)
            self.coords = point_cloud[:, :3]
            self.occupancies = point_cloud[:, 3]

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):

        length = self.coords.shape[0]
        idx_size = self.on_surface_points
        idx = np.random.randint(length, size=idx_size)

        coords = self.coords[idx]
        occs = self.occupancies[idx, None]

        return {"coords": torch.from_numpy(coords).float()}, {
            "sdf": torch.from_numpy(occs)
        }


def get_pc(
    files: list = [file for file in os.listdir("./datasets/02691156_watertight_mesh")],
):

    for i, file in tqdm(enumerate(files), "File"):
        PointCloud(
            "./datasets/02691156_watertight_mesh/" + files[i], 2048, strategy="save_pc"
        )
        print(f"Done training file: {files[i]}")


def plot_image(
    path: str = "./datasets/02691156_pc/f6e6bab105ab890080be894a49cc4571.obj",
):
    dataset = PointCloud(path, 2048, strategy="")
    import matplotlib.pyplot as plt

    coords, sdf = (
        torch.from_numpy(dataset.coords),
        torch.from_numpy(dataset.occupancies),
    )
    # Get the indices where sdf values are <= 0
    indices_in = torch.where(sdf == -1)[0]
    indices_out = torch.where(sdf == 1)[0]

    filtered_coord_in = coords[indices_in]
    filtered_coord_out = coords[indices_out]

    mesh = trimesh.load(
        "./datasets/02691156_watertight_mesh/"
        + path.split("/")[-1].split(".")[0]
        + ".obj"
    )
    print("filtered_coord_out.shape: ", filtered_coord_out.shape)
    print("filtered_coord_in.shape: ", filtered_coord_in.shape)

    # Extract x, y, z coordinates
    x_in = filtered_coord_in[:20000, 0].numpy()
    y_in = filtered_coord_in[:20000, 1].numpy()
    z_in = filtered_coord_in[:20000, 2].numpy()

    x_out = torch.cat(
        (filtered_coord_out[:1000, 0], filtered_coord_out[-20000:, 0])
    ).numpy()
    y_out = torch.cat(
        (filtered_coord_out[:1000, 1], filtered_coord_out[-20000:, 1])
    ).numpy()
    z_out = torch.cat(
        (filtered_coord_out[:1000, 2], filtered_coord_out[-20000:, 2])
    ).numpy()

    print(x_out.shape)

    plt.rcParams["text.usetex"] = True

    # Create a scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(x_in, z_in, y_in, c="green", s=0.1)
    scatter = ax.scatter(x_out, z_out, y_out, c="blue", s=0.05)

    scatter = ax.scatter([], [], [], c="green", s=1000, label=r"$SDF(x) < 0$")
    scatter = ax.scatter([], [], [], c="blue", s=1000, label=r"$SDF(x) > 0$")

    # Set labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xlim((-0.5, 0.5))
    ax.set_ylim((-0.5, 0.5))
    ax.set_zlim((-0.5, 0.5))

    # Show plot
    plt.grid(False)
    plt.axis("off")
    plt.legend(fontsize="30", loc="lower left")
    plt.show()


def main():
    """
    Expects *watertight* shapenet meshes to be present in ./datasets/02691156_watertight_mesh
    """
    get_pc()


if __name__ == "__main__":
    main()
