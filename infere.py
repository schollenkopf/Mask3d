print("imports")
from mask3d import (
    get_model,
    load_mesh,
    prepare_data,
    map_output_to_pointcloud,
    save_colorized_mesh,
    get_lists,
)
import torch
import os
import numpy as np
print("getting model")
model = get_model("checkpoints/last-epoch830its.ckpt", "s3dis")
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
    confidences, instances_mapped_list, labels_mapped_list = get_lists(mesh,outputs,inverse_map)

    np.save("scans/confidences_"+os.path.splitext(filename)[0]+".npy",np.array(confidences),allow_pickle=True)
    np.save("scans/instances_"+os.path.splitext(filename)[0]+".npy",np.array(instances_mapped_list),allow_pickle=True)
    np.save("scans/labels_"+os.path.splitext(filename)[0]+".npy",np.array(labels_mapped_list),allow_pickle=True)


    # labels, instances = map_output_to_pointcloud(mesh, outputs, inverse_map)

    # save_path_instance = os.path.join(
    #     "scans/", "instance_" + os.path.splitext(filename)[0] + ".ply"
    # )
    # save_colorized_mesh(mesh, instances, save_path_instance, labelMode=False)
    # save_path_label = os.path.join(
    #     "scans/", "labeled_" + os.path.splitext(filename)[0] + ".ply"
    # )
    # save_colorized_mesh(mesh, labels, save_path_label)
