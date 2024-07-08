from torch.utils.data import Dataset
import trimesh
import igl
import numpy as np
import os
import torch


class PointCloud(Dataset):
    def __init__(
        self,
        path,
        on_surface_points,
        keep_aspect_ratio=True,
        is_mesh=True,
        output_type="occ",
        out_act="sigmoid",
        n_points=100000,
        strategy="save_pc",
        cfg=None,
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
                vertices = obj.vertices
                
                vertices -= np.mean(vertices, axis=0, keepdims=True)
                v_max = np.amax(vertices)
                v_min = np.amin(vertices)
                vertices *= 0.5 * 0.95 / (max(abs(v_min), abs(v_max)))
                obj.vertices = vertices
                self.obj = obj
                
                
                total_points = n_points  # 100000
                n_points_uniform = total_points  # int(total_points * 0.5)
                n_points_surface = total_points  # total_points

                points_uniform = np.random.uniform(
                    -0.5, 0.5, size=(n_points_uniform, 3)
                )
                points_surface = obj.sample(n_points_surface)
                points_surface += 0.01 * np.random.randn(n_points_surface, 3)
                points = np.concatenate([points_surface, points_uniform], axis=0)
                
                trimesh.repair.fix_winding(obj)
                trimesh.repair.fix_inversion(obj)
                trimesh.repair.fill_holes(obj)
                
                inside_surface_values = igl.fast_winding_number_for_meshes(
                    obj.vertices, obj.faces, points
                )
                inside_surface_values = igl.signed_distance(points, obj.vertices, obj.faces)[0]
                
                
                #obj.show()
                
                # for debugging 
                self.points = points
                self.inside_surface_values = inside_surface_values
                
                thresh = 0.5
                occupancies_winding = np.piecewise(
                    inside_surface_values,
                    [inside_surface_values < thresh, inside_surface_values >= thresh],
                    [0, 1],
                )
                
                occupancies = occupancies_winding[..., None]
                print(points.shape, occupancies.shape, occupancies.sum())
                point_cloud = points
                point_cloud = np.hstack((point_cloud, occupancies))
                print(point_cloud.shape, points.shape, occupancies.shape)

        else:
            point_cloud = np.genfromtxt(path)
        print("Finished loading point cloud")
        
        pc_folder ="./datasets/02691156_pc"

        
        if strategy == "save_pc":
            self.coords = point_cloud[:, :3]
            self.normals = point_cloud[:, 3:]

            point_cloud_xyz = np.hstack((self.coords, self.normals))
            os.makedirs(pc_folder, exist_ok=True)
            np.save(pc_folder + "/" + path.split("/")[-1].split(".")[0], point_cloud_xyz)
        else:
            point_cloud = np.load(
                pc_folder + "/" + path.split("/")[-1].split(".")[0] + ".npy"
            )
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
        
if __name__ == "__main__":
    files = [
        file
        for file in os.listdir("./datasets/02691156_watertight_mesh")
    ]
    for i, file in enumerate(files):
        dataset = PointCloud("./datasets/02691156_watertight_mesh/" + files[5], 2048, strategy="save_pc")
        break
        
    import matplotlib.pyplot as plt

    
    sdf = torch.from_numpy(dataset.inside_surface_values)
    coords = torch.from_numpy(dataset.points)
    
    # Get the indices where sdf values are <= 0
    indices = torch.where(sdf < 0.0)[0]

    filtered_coord = coords[indices]
        

    # Extract x, y, z coordinates
    x = filtered_coord[:, 0].numpy()
    y = filtered_coord[:, 1].numpy()
    z = filtered_coord[:, 2].numpy()

    # Create a scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, z, y, s=1)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim((-0.5, 0.5))
    ax.set_ylim((-0.5, 0.5))
    ax.set_zlim((-0.5, 0.5))

    # Show plot
    plt.show()
        
