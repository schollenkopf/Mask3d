import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from scipy.spatial.transform import Rotation as R

colormap = {"wall":[1, 185/255, 185/255], "column": [1.0, 0.55, 0.0], "clutter": [1, 0,1], "board": [0.0, 0.0, 1.0]}

radius = np.random.uniform(1.8, 3)



def generate_cylinder_wall(radius=radius, height=10.0, num_points=1000000):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(-height/2, height/2, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.vstack((x, y, z)).T


def generate_ladder_with_noise(
    radius=radius,
    height=10,
    num_noise_points=1000,
    rail_thickness=0.025,
    rung_thickness=0.025,
    radial_std=0.2,
    thickness_resolution=25
):
    clearance = np.random.uniform(0.2, 0.5)
    rung_spacing = np.random.uniform(0.25, 0.45)
    rail_width = rung_spacing + np.random.uniform(0.1, 0.2)

    angle = np.random.uniform(0, 2 * np.pi)
    ladder_distance = radius - clearance
    base = np.array([ladder_distance * np.cos(angle), ladder_distance * np.sin(angle), 0])

    direction = np.array([-np.sin(angle), np.cos(angle), 0])  # Tangential to circle
    up = np.array([0, 0, 1])
    normal = np.cross(direction, up)  # Outward from ladder

    ladder_points = []

    def add_thick_point(center, thickness):
        for theta in np.linspace(0, 2 * np.pi, thickness_resolution, endpoint=False):
            offset = np.cos(theta) * normal + np.sin(theta) * up
            ladder_points.append(center + offset * thickness)

    # Vertical rails (2 sides)
    for offset in [0, rail_width]:
        rail_base = base + direction * offset
        for z in np.linspace(0, height, 200):
            center = rail_base + np.array([0, 0, z - height / 2])
            add_thick_point(center, rail_thickness)

    # Rungs (horizontal bars)
    num_rungs = int(height / rung_spacing)
    for i in range(num_rungs):
        z = i * rung_spacing - height / 2
        for t in np.linspace(0, 1, 50):
            center = base + direction * (t * rail_width) + np.array([0, 0, z])
            add_thick_point(center, rung_thickness)

    # Generate noise around full ladder volume
    theta = np.random.uniform(0, 2 * np.pi, num_noise_points)
    r = np.random.normal(0, radial_std, num_noise_points)
    z = np.random.uniform(-height / 2, height / 2, num_noise_points)

    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    dz = z

    noise = (
        base + direction*rail_width/2
        + np.outer(dx, normal)
        + np.outer(dy, direction)
        + np.outer(dz, up)
    )

    return np.array(ladder_points), noise


def generate_random_noise_points(num_points=500, max_radius=radius, max_height=10.0):
    r = np.sqrt(np.random.uniform(0, max_radius**2, num_points))
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(-max_height/2, max_height/2, num_points)
    return np.vstack((x, y, z)).T


def generate_vertical_cable(height=10):

    num_circ = 100
    num_height = 500
    radius = np.random.uniform(0.01, 0.2)
    center_angle = np.random.uniform(0, 2 * np.pi)
    radial_dist = np.random.uniform(0.5, 2.3)
    theta = np.linspace(0, 2 * np.pi, num_circ)
    z = np.linspace(-height/2, height/2, num_height)
    t, zz = np.meshgrid(theta, z)

    x_center = radial_dist * np.cos(center_angle)
    y_center = radial_dist * np.sin(center_angle)
    x = x_center + radius * np.cos(t)
    y = y_center + radius * np.sin(t)

    cable_points = np.vstack((x.flatten(), y.flatten(), zz.flatten())).T
    return cable_points


def random_lidar_position(max_radius=radius - 0.8):
    r = np.sqrt(np.random.uniform(0, max_radius**2))  # uniform in area
    theta = np.random.uniform(0, 2 * np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = 0  # np.random.uniform(0.5, height - 0.5)
    return np.array([x, y, z])


def fuzzy_elliptical_block_mask(num_lines, num_azimuths, seed=None):
    """
    Generates a drone occlusion mask shaped like a noisy ellipse,
    with soft fuzzy edges to simulate partial occlusion.

    Parameters:
        num_lines (int): Vertical resolution of LiDAR.
        num_azimuths (int): Horizontal resolution of LiDAR.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Boolean mask of shape (num_lines, num_azimuths).
    """
    if seed is not None:
        np.random.seed(seed)

    mask = np.zeros((num_lines, num_azimuths), dtype=bool)

    center_el = 10
    center_az = num_azimuths // 2

    radius_el = num_lines // 3
    radius_az = num_azimuths // 5

    for i in range(num_lines):
        for j in range(num_azimuths):
            dy = (i - center_el) / radius_el
            dx = (j - center_az) / radius_az
            dist2 = dx**2 + dy**2

            # Inner core: fully blocked
            if dist2 <= 0.2:
                mask[i, j] = True

            # Border zone: probabilistic dropoff
            elif dist2 <= 1.2:
                prob = np.exp(-(dist2 - 0.2) * 3)
                if np.random.rand() < prob:
                    mask[i, j] = True

    return mask


def simulate_lidar_grid(
    points,
    lidar_pos,
    lidar_rot=R.from_euler('y', 20, degrees=True).as_matrix(),
    num_lines=40,
    block_mask_fn=None,
):
    """
    Simulate a LiDAR scan by projecting 3D points into a spherical grid from the LiDAR's position and orientation.
    Applies occlusion masking and returns a visibility mask over the original points.
    
    Args:
        points (np.ndarray): Nx3 array of world points.
        lidar_pos (np.ndarray): 3D position of the LiDAR.
        lidar_rot (np.ndarray): 3x3 rotation matrix (LiDAR orientation).
        num_lines (int): Number of vertical scan lines (LiDAR channels).
        block_mask_fn (callable): Optional function to apply occlusion in the LiDAR grid.

    Returns:
        visible_mask (np.ndarray): Boolean array of shape (N,), True for points seen by the LiDAR.
    """

    num_lines = np.random.randint(40, 64)
    num_azimuths = np.random.randint(400, 600)

    vertical_fov = (
        np.random.randint(-20, -10) * np.pi / 180,
        np.random.randint(45, 55)* np.pi / 180
    )

    # Transform points into the LiDAR frame
    rel_points = points - lidar_pos
    rel_points_lidar = (lidar_rot.T @ rel_points.T).T
    rel_dists = np.linalg.norm(rel_points_lidar, axis=1)
    rel_dirs = rel_points_lidar / (rel_dists[:, np.newaxis] + 1e-8)
    # Compute spherical coordinates (azimuth and elevation)
    az = np.arctan2(rel_dirs[:, 1], rel_dirs[:, 0])
    el = np.arcsin(rel_dirs[:, 2])
    az[az < 0] += 2 * np.pi  # wrap azimuth

    # Discretize into LiDAR grid
    az_idx = (az / (2 * np.pi) * num_azimuths).astype(int)
    el_idx = ((el - vertical_fov[0]) / (vertical_fov[1] - vertical_fov[0]) * num_lines).astype(int)

    valid = (
        (el_idx >= 0) & (el_idx < num_lines) &
        (az_idx >= 0) & (az_idx < num_azimuths)
    )

    # Filter valid points
    az_idx = az_idx[valid]
    el_idx = el_idx[valid]
    rel_dists = rel_dists[valid]
    rel_points_lidar = rel_points_lidar[valid]
    orig_indices = np.nonzero(valid)[0]

    # keep closest point for each angle pair (ray direction)
    lidar_grid = np.full((num_lines, num_azimuths), np.nan)
    distance_grid = np.full((num_lines, num_azimuths), np.inf)
    for i in range(len(az_idx)):
        r_, c_ = el_idx[i], az_idx[i]
        if rel_dists[i] < distance_grid[r_, c_]:
            lidar_grid[r_, c_] = orig_indices[i]
            distance_grid[r_, c_] = rel_dists[i]
    
    # Simulate occlusion from drone body
    if block_mask_fn:
        block_mask = block_mask_fn(num_lines, num_azimuths)
        lidar_grid[block_mask] = np.nan

    # random dropout
    row_indices = np.arange(num_lines).reshape(-1, 1)
    dropout_probs = 0.02 * (1 - np.abs((row_indices - num_lines / 2) / (num_lines / 2)))
    dropout_mask = np.random.rand(num_lines, num_azimuths) < dropout_probs
    lidar_grid[dropout_mask] = np.nan


    # Extract visible points from the grid
    flat_points = lidar_grid.reshape(-1)
    valid_cells = ~np.isnan(flat_points[:])
    selected_points = flat_points[valid_cells]
    valid_mask = np.zeros(points.shape[0], dtype=bool)
    for idx in selected_points:
        valid_mask[int(idx)] = True
    return valid_mask

def make_measurements_noisy(points, std_dev=0.008):
    return points + np.random.normal(0, std_dev, points.shape)

def save_dataset_as_s3dis(num_samples=600):
    for folder, num in zip(
        [
            # "Area_1",
            "Area_2",
            # "Area_3",
            # "Area_4",
            # "Area_5",
            # "Area_6",
        ],
        [20]
        # [num_samples, 1, 1, 1, 1, 1],
    ):
        for n in range(int(num)):
            if n % 10 == 0:
                print(n / 500)
            sample_name = "sample_" + str(n + 1)
            path = "dataset/" + folder + "/" + sample_name
            annotation_path = path + "/Annotations"
            os.makedirs(path, exist_ok=True)
            os.makedirs(annotation_path, exist_ok=True)
            instances = []
            cylinder_points = generate_cylinder_wall()
            instances += [(cylinder_points, "wall")]
            ladder_points, ladder_noise_points = generate_ladder_with_noise()
            instances += [(ladder_points, "board")]
            noise_points = generate_random_noise_points()

            instances += [(np.vstack([ladder_noise_points, noise_points]), "clutter")]

            all_cable_points = []
            cable_count = np.random.randint(1, 4)
            for i in range(cable_count):
                cable_points = generate_vertical_cable()
                instances += [(cable_points, "column")]
                all_cable_points.append(cable_points)

            all_cable_points = np.vstack(all_cable_points)

            scene_points_pre_perspective = np.vstack(
                (
                    cylinder_points,
                    ladder_points,
                    ladder_noise_points,
                    noise_points,
                    all_cable_points,
                )
            )

            lidar_pos = random_lidar_position()
            rot = R.from_euler('X', np.random.randint(-25,25), degrees=True).as_matrix()
            valid_mask = simulate_lidar_grid(scene_points_pre_perspective, lidar_pos,lidar_rot=rot,block_mask_fn=fuzzy_elliptical_block_mask)
            scene_points_final = []
            pcds = []
            for instance_points, instance_class in instances:
                instance_points = make_measurements_noisy(instance_points)
                number_instance_points = len(instance_points)
                instance_valid_mask = valid_mask[:number_instance_points]
                valid_instance_points = instance_points[instance_valid_mask]
                # valid_instance_points = (rot @ valid_instance_points.T).T
                valid_mask = valid_mask[number_instance_points:]
                colors = np.zeros((valid_instance_points.shape[0], 3))
                valid_instance_points_with_color = np.hstack(
                    (valid_instance_points, colors)
                )

                filename = (
                    instance_class + "_1.txt"
                    if instance_class != "column"
                    else instance_class + "_" + str(cable_count) + ".txt"
                )
                if not len(valid_instance_points_with_color) < 10:
                    np.savetxt(
                        annotation_path + "/" + filename,
                        valid_instance_points_with_color,
                        fmt="%.6f",
                        delimiter=" ",
                    )

                    cable_count = (
                        cable_count if instance_class != "column" else cable_count - 1
                    )
                    scene_points_final.append(valid_instance_points_with_color)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(valid_instance_points_with_color[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector( np.tile(colormap[instance_class], (len(pcd.points), 1)))
                    pcds += [pcd]
            scene_points_final = np.vstack(scene_points_final)
            np.savetxt(
                path + "/" + sample_name + ".txt",
                scene_points_final,
                fmt="%.6f",
                delimiter=" ",
            )
            # pcd2 = o3d.io.read_point_cloud("scans/source.ply")
            # pcd = o3d.geometry.PointCloud()
            
            # pcd.points = o3d.utility.Vector3dVector(scene_points_final[:, :3])
            # pcd.colors = o3d.utility.Vector3dVector(scene_points_final[:, 3:])
            # pcd2.colors = o3d.utility.Vector3dVector(
            #     np.tile([0.5, 0.0, 0.0], (len(pcd2.points), 1))
            # )
            o3d.visualization.draw_geometries([*pcds])
            # o3d.io.write_point_cloud("test.ply", pcd)


if __name__ == "__main__":
    save_dataset_as_s3dis()

# import matplotlib.pyplot as plt
# import numpy as np

# colors = np.array([
#     [0, 0, 1], [0, 0, 0.5], [0, 1, 0], [0, 1, 1], [1, 0, 1],
#     [0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [0.5, 0.5, 1],
#     [0, 0.5, 0], [0.5, 0, 0], [1, 0, 0], [0.5, 0, 1]
# ])
# labels = [
#     "ceiling", "floor", "cylinder", "beam", "cable", "window",
#     "door", "table", "chair", "sofa", "bookcase", "ladder", "noise"
# ]

# plt.figure(figsize=(6, 2))
# for i, (color, label) in enumerate(zip(colors, labels)):
#     plt.bar(i, 1, color=color)
#     plt.text(i, 1.05, label, ha='center', va='bottom', rotation=90, fontsize=8)

# plt.xticks([])
# plt.yticks([])
# plt.box(False)
# plt.tight_layout()
# plt.show()