print("imports")
from mask3d import (
    get_model,
    load_mesh,
    prepare_data,
    map_output_to_pointcloud,
    save_colorized_mesh,
)
import torch
import os

print("getting model")
model = get_model("checkpoints/last-epoch.ckpt", "s3dis")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("loading mesh")
# load input data
for i, filename in enumerate(os.listdir("scans/")):
    print("Handling: ", str(i))
    pointcloud_file = os.path.join("scans/", filename)
    mesh = load_mesh(pointcloud_file)

    # prepare data
    data, points, raw_coordinates, colors, features, unique_map, inverse_map = (
        prepare_data(mesh, device)
    )

    with torch.no_grad():
        outputs = model(data, raw_coordinates=raw_coordinates)

    labels = map_output_to_pointcloud(mesh, outputs, inverse_map)

    save_path_instance = os.path.join(
        "scans/", "instance_" + os.path.splitext(filename)[0] + ".ply"
    )
    save_colorized_mesh(mesh, labels, save_path_instance, label=False)
    save_path_label = os.path.join(
        "scans/", "labeled_" + os.path.splitext(filename)[0] + ".ply"
    )
    save_colorized_mesh(mesh, labels, save_path_label)
