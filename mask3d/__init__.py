import hydra
import torch

from mask3d.models.mask3d import Mask3D
from mask3d.utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)


class InstanceSegmentation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = hydra.utils.instantiate(cfg.model)

    def forward(self, x, raw_coordinates=None):
        return self.model(x, raw_coordinates=raw_coordinates)


from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import initialize, compose

# imports for input loading
import albumentations as A
import MinkowskiEngine as ME
import numpy as np
from sklearn.cluster import DBSCAN
import open3d as o3d

# imports for output
from mask3d.datasets.scannet200.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
    SCANNET_COLOR_MAP_20,
    SCANNET_COLOR_MAP_200,
)


def get_model(checkpoint_path=None, dataset_name="scannet200"):

    # Initialize the directory with config files
    with initialize(config_path="conf"):
        # Compose a configuration
        cfg = compose(config_name="config_base_instance_segmentation.yaml")

    cfg.general.checkpoint = checkpoint_path

    # would be nicd to avoid this hardcoding below
    # dataset_name = checkpoint_path.split("/")[-1].split("_")[0]
    if dataset_name == "scannet200":
        cfg.general.num_targets = 201
        cfg.general.train_mode = False
        cfg.general.eval_on_segments = True
        cfg.general.topk_per_image = 300
        cfg.general.use_dbscan = True
        cfg.general.dbscan_eps = 0.95
        cfg.general.export_threshold = 0.001

        # # data
        cfg.data.num_labels = 200
        cfg.data.test_mode = "test"

        # # model
        cfg.model.num_queries = 150
    if dataset_name == "s3dis":
        cfg.general.num_targets = 5
        cfg.general.train_mode = False
        cfg.general.eval_on_segments = True
        cfg.general.topk_per_image = 100
        cfg.general.use_dbscan = True
        cfg.general.dbscan_eps = 0.95
        cfg.general.export_threshold = 0.00001

        cfg.data.num_labels = 4
        cfg.data.add_colors = False
        cfg.data.add_normals = False
        cfg.data.in_channels = 3
        cfg.data.test_mode = "test"

        cfg.model.num_queries = 50

    if dataset_name == "scannet":
        cfg.general.num_targets = 19
        cfg.general.train_mode = False
        cfg.general.eval_on_segments = True
        cfg.general.topk_per_image = 300
        cfg.general.use_dbscan = True
        cfg.general.dbscan_eps = 0.95
        cfg.general.export_threshold = 0.001

        # # data
        cfg.data.num_labels = 20
        cfg.data.test_mode = "test"

        # # model
        cfg.model.num_queries = 150

        # TODO: this has to be fixed and discussed with Jonas
        # cfg.model.scene_min = -3.
        # cfg.model.scene_max = 3.

    # # Initialize the Hydra context
    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # hydra.initialize(config_path="conf")

    # Load the configuration
    # cfg = hydra.compose(config_name="config_base_instance_segmentation.yaml")
    model = InstanceSegmentation(cfg)

    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return model


def load_mesh(pcl_file):

    # load point cloud
    input_mesh_path = pcl_file
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    return mesh


def prepare_data(mesh, device):

    coordinates = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    colors = np.ones((len(colors), 3))
    colors = colors * 255.0

    points = coordinates.copy()
    features = np.hstack((colors, coordinates))

    # voxelise with 2cm resolution
    coords = np.floor(coordinates / 0.02)
    _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=torch.from_numpy(coords).to("cpu").contiguous(),
        features=colors,
        return_index=True,
        return_inverse=True,
    )

    sample_coordinates = coords[unique_map]
    coordinates = [
        torch.from_numpy(sample_coordinates).int()
    ]  # [] bc its a batch of size 1
    sample_features = features[unique_map]
    features = [
        torch.from_numpy(sample_features).float()
    ]  # [] bc its a batch of size 1

    coordinates, features = ME.utils.sparse_collate(coords=coordinates, feats=features)

    raw_coordinates = features[:, -3:]
    features = features[:, :-3]
    # features = torch.cat(features, dim=0)
    data = ME.SparseTensor(
        coordinates=coordinates,
        features=features,
        device=device,
    )

    return (
        data,
        points,
        raw_coordinates,
        colors,
        features,
        unique_map,
        inverse_map,
    )

def get_lists(
    mesh, outputs, inverse_map
):

    # parse predictions
    logits = outputs["pred_logits"]
    masks = outputs["pred_masks"]

    # reformat predictions
    logits = logits[0].detach().cpu()
    masks = masks[0].detach().cpu()
    labels = []
    confidences = []
    instance_masks = []
    for i in range(len(logits)):
        p_labels = torch.softmax(logits[i], dim=-1)
        p_masks = torch.sigmoid(masks[:, i])
        l = torch.argmax(p_labels, dim=-1)
        c_label = torch.max(p_labels)
        m = p_masks > 0.5
        c_m = p_masks[m].sum() / (m.sum() + 1e-8)
        c = c_label * c_m
        if l < 200: # and c > confidence_threshold:
            labels.append(int(l.item()))
            confidences.append(float(c.item()))
            instance_masks.append(m[inverse_map])
    
    final_instances = np.zeros((len(mesh.vertices), 1))
    final_labels = np.zeros((len(mesh.vertices), 1))
    foundLadder = False
    instance_id = 1 
    for (conf,label, m) in sorted(
            zip(confidences, labels, instance_masks),
            key=lambda x: (x[0], x[1]),
            reverse=True,
        ):
        if int(label) == 1 and conf > 0.85:
            pass
        elif int(label) == 2 and not foundLadder and conf > 0.75:
            foundLadder = True
            pass
        elif int(label)!=4 and conf > 0.85:
            pass
        else:
            continue
        final_instances[m == 1] = instance_id
        final_labels[m == 1] = label + 1
        instance_id += 1

    return final_instances, final_labels


def map_output_to_pointcloud(
    mesh, outputs, inverse_map, label_space="scannet200", confidence_threshold=0.9
):

    # parse predictions
    logits = outputs["pred_logits"]
    masks = outputs["pred_masks"]

    # reformat predictions
    logits = logits[0].detach().cpu()
    masks = masks[0].detach().cpu()

    labels = []
    confidences = []
    instance_masks = []

    for i in range(len(logits)):
        p_labels = torch.softmax(logits[i], dim=-1)
        p_masks = torch.sigmoid(masks[:, i])
        l = torch.argmax(p_labels, dim=-1)
        c_label = torch.max(p_labels)
        m = p_masks > 0.5
        c_m = p_masks[m].sum() / (m.sum() + 1e-8)
        c = c_label * c_m


        if l < 200  and c > confidence_threshold:
            # full_res_mask = m[inverse_map]
            # masked_coords = np.asarray(mesh.vertices)[full_res_mask]

            # if len(masked_coords) == 0:
            #     continue

            # # Apply DBSCAN
            # db = DBSCAN(eps=0.95, min_samples=1).fit(masked_coords)
            # cluster_labels = db.labels_

            # for cluster_id in np.unique(cluster_labels):
            #     if cluster_id == -1 or cluster_id == 13:
            #         continue  # skip noise

            #     cluster_mask = cluster_labels == cluster_id
            #     cluster_indices = np.where(full_res_mask)[0][cluster_mask]

            #     # Create binary mask for this cluster
            #     instance_mask = torch.zeros(len(mesh.vertices), dtype=torch.bool)
            #     instance_mask[cluster_indices] = True

            labels.append(int(l.item()))
            confidences.append(float(c.item()))
            instance_masks.append(m[inverse_map])


    labels_mapped = np.zeros((len(mesh.vertices), 1))
    instances_mapped = np.zeros((len(mesh.vertices), 1))
    instance_id = 1  # Starting instance ID

    for i, (c,l, m) in enumerate(
        sorted(
            zip(confidences, labels, instance_masks),
            key=lambda x: (x[0], x[1]),
            reverse=False,
        )
    ):
        print(l)
        # if label_space == "scannet200":
        #     label_offset = 2
        #     if l == 0:
        #         l = -1 + label_offset
        #     else:
        #         l = int(l) + label_offset

        instances_mapped[m == 1] = instance_id
        labels_mapped[m == 1] = l
        instance_id += 1

    return labels_mapped, instances_mapped


def save_colorized_mesh(mesh, labels_mapped, output_file, labelMode=True):
    colors = np.array(
        [
            [0, 1, 0],  # "wall",
            [1, 0, 1],  # "column",
            [1, 0, 0],  # "board",
            [0.5, 0, 1],  # "clutter",
            [1, 1, 1],  # "inactivate",
        ]
    )
    unique_labels = np.unique(labels_mapped)
    np.random.seed(42)  # For reproducibility
    if not labelMode:
        colors = np.random.rand(len(unique_labels), 3)

    # Map colors to the mesh
    vertex_colors = np.zeros((len(mesh.vertices), 3))
    for i, label in enumerate(unique_labels):
        if labelMode:
            vertex_colors[labels_mapped[:, 0] == label] = colors[i]
        else:
            vertex_colors[labels_mapped[:, 0] == label] = colors[i]

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    o3d.io.write_triangle_mesh(output_file, mesh)


if __name__ == "__main__":

    model = get_model("checkpoints/scannet200/scannet200_benchmark.ckpt")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load input data
    pointcloud_file = "data/pcl.ply"
    mesh = load_mesh(pointcloud_file)

    # prepare data
    data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)

    # run model
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)

    # map output to point cloud
    labels = map_output_to_pointcloud(mesh, outputs, inverse_map)

    # save colorized mesh
    save_colorized_mesh(mesh, labels, "data/pcl_labelled.ply", colormap="scannet200")
