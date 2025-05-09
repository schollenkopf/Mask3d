print("imports")
from mask3d import (
    get_model,
    load_mesh,
    prepare_data,
    map_output_to_pointcloud,
    save_colorized_mesh,
)
import torch

print("getting model")
model = get_model("checkpoints/area4_from_scratch.ckpt", "s3dis")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("loading mesh")
# load input data
pointcloud_file = "scans/1717333010226698.ply "
mesh = load_mesh(pointcloud_file)

# prepare data
print("prepare mesh")
data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)

# run model
print("run model")
with torch.no_grad():
    outputs = model(data, raw_coordinates=features)

print("map output")
# map output to point cloud
labels = map_output_to_pointcloud(mesh, outputs, inverse_map)

print("color cloud")
# save colorized mesh
save_colorized_mesh(mesh, labels, "scans/labeled.ply", colormap="scannet200")
