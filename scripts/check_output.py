import numpy as np
import open3d as o3d
import os
fileName = "source"
export = False
#0:cylinder, 1: cable,2: ladder, 3: noise, 4: inactivate
classes = ["cylinder", "cable","ladder", "noise", "inactive"]
classesColors = {"cylinder":np.array([1, 185/255, 185/255]),"cable":np.array([1.0, 0.55, 0.0]),"ladder":np.array([0.0, 0.0, 1.0]), "noise":np.array([1, 0,1]),"inactive":np.array([0,0,0])}
cableColors = [np.array([0,1,0]),np.array([0.2,0.8,0.2]),np.array([0.4,0.6,0.4]),np.array([0,0.4,0.6]),np.array([0,0.2,0.8])]
instanceColors = [
    np.array([0.121, 0.466, 0.705]),  # blue
    np.array([1.000, 0.498, 0.054]),  # orange
    np.array([0.172, 0.627, 0.172]),  # green
    np.array([0.839, 0.153, 0.157]),  # red
    np.array([0.580, 0.404, 0.741]),  # purple
    np.array([0.549, 0.337, 0.294]),  # brown
    np.array([0.890, 0.467, 0.761]),  # pink
    np.array([0.498, 0.498, 0.498])   # gray
]
# for i, filename in enumerate(os.listdir("data/830_its_out/")):
#     if os.path.splitext(filename)[-1] == ".ply":
        # fileName = os.path.splitext(filename)[0]
# confidences = np.load(f"data/830_its_out/confidences_{fileName}.npy")
# instance_masks_list = np.load(f"data/830_its_out/instances_{fileName}.npy")
# label_masks_list = np.load(f"data/830_its_out/labels_{fileName}.npy")
# labels = [classes[int(np.unique(l)[-1])] for l in label_masks_list]
# pcd = o3d.io.read_point_cloud(f"data/830_its_out/{fileName}.ply")
# label = 0

# colors = np.zeros((len(pcd.points), 3))
# cableInstanceCount = 0
# totalInstanceCount = 1
# foundLadder = False
# final_instances = np.zeros(len(pcd.points))
# final_labels = np.zeros(len(pcd.points))
# for (conf,label,labmask,insmask) in sorted(
#             zip(confidences, labels,label_masks_list, instance_masks_list),
#             key=lambda x: (x[0], x[1]),
#             reverse=True,
#         ):
#     if label == "cable" and conf > 0.9:
#         colors[insmask[:, 0] != 0] = cableColors[cableInstanceCount]
#         cableInstanceCount += 1
#     elif label == "ladder" and not foundLadder and conf > 0.8:
#         print(conf)
#         colors[insmask[:, 0] != 0] = classesColors[label]
#         foundLadder = True
#     elif label!="inactive" and conf > 0.9:
#         colors[insmask[:, 0] != 0] = classesColors[label]
#     else:
#         continue
#     final_instances[insmask[:, 0] != 0] = totalInstanceCount
#     final_labels[insmask[:, 0] != 0] = classes.index(label) + 1
#     totalInstanceCount += 1
#     if export:
#         mask = insmask[:, 0] != 0
#         # Extract points and optionally colors belonging to this instance
#         instance_points = np.asarray(pcd.points)[mask]
#         instance_colors = colors[mask]
#         exportPCD = o3d.geometry.PointCloud()
#         exportPCD.points = o3d.utility.Vector3dVector(instance_points)
#         exportPCD.colors = o3d.utility.Vector3dVector(instance_colors)
#         cableLabel = str(cableInstanceCount) if label == "cable" else ""
#         o3d.io.write_point_cloud(
#             f"data/830_its_out/{fileName}/{fileName}_{label}{cableLabel}.ply", exportPCD,write_ascii=True
#         )
# np.save("test.npy",np.array(final_labels),allow_pickle=True)



def pca_downsample(pcd, threshold=0.1, downsample_factor=20,voxelsize = 0.15):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
    )
    # o3d.visualization.draw_geometries(
    #     [pcd], point_show_normal=True, window_name="target and source"
    # )
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    translation_C = normals.T @ normals
    eigenvalues, eigenvectors = np.linalg.eigh(translation_C)

    sorted_indices = np.argsort(eigenvalues)

    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    scaled_eigenvalues = eigenvalues / np.sum(eigenvalues)
    # print(scaled_eigenvalues)
    # print(scaled_eigenvalues)
    if scaled_eigenvalues[2] > (1 - threshold):
        # Flat surface (e.g., wall or floor): one dominant direction
        weak_axes = eigenvectors[:, :2]
    elif scaled_eigenvalues[0] < threshold:
        # Linear structure (e.g., column): one weak direction
        weak_axes = eigenvectors[:, :1]
    else:
        # no need to downsample
        return points, points

    source_points = points


    # for axis in weak_axes.T:
    #     if (len(source_points)> 0):
    #         projections = source_points @ axis
    #         low = projections.min()
    #         high = projections.max()
    #         lower_threshold = low + (high - low) * (1/6)
    #         upper_threshold = high - (high - low) * (1/6)

    #         mask = (projections >= lower_threshold) & (projections <= upper_threshold)
    #         source_points = source_points[mask]
    for axis in weak_axes.T:
        projections = source_points @ axis
        low = np.percentile(projections, 25)
        high = np.percentile(projections, 75)
        mask = (low <= projections ) & (projections <= high)
        source_points = source_points[mask]


    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(points)

    map_pcd = map_pcd.voxel_down_sample(voxel_size=voxelsize*2.5)
    source_pcd = source_pcd.voxel_down_sample(voxel_size=voxelsize*2)

    return np.asarray(source_pcd.points), np.asarray(map_pcd.points)
    


stats = {}
skip = 5
skipped_Frames = 0
for i, filename in enumerate(sorted(os.listdir("data/esbjerg/scansSubSample/"))):
    
    if os.path.splitext(filename)[-1] == ".ply" and os.path.splitext(filename)[0] == "1741440573788945":# and i>910 and i%skip == 0:
        print(i)
        instances = []


        fileName = os.path.splitext(filename)[0]
        pcd = o3d.io.read_point_cloud(f"data/esbjerg/scansSubSample/{fileName}.ply")

        final_instances = np.load(f"data/esbjerg/model_outputs/instances_{fileName}.npy",allow_pickle = True)
        final_labels = np.load(f"data/esbjerg/model_outputs/labels_{fileName}.npy",allow_pickle = True)
        print(np.unique(final_instances))

        

        instance_ids = np.unique(final_instances)
        map_geometries = []
        source_geometries = []
        source_colors = []
        map_colors = []
        nr_ladder_points = 0
        nr_instances_labels = {1:0,2:0,3:0,4:0,5:0}
        for instance_id in instance_ids:
            mask = final_instances == instance_id
            label = int(final_labels[mask][0])
            if instance_id!=0:
                nr_instances_labels[label] += 1
            o3d.io.write_point_cloud(f"data/target_{instance_id}.ply", pcd.select_by_index(np.where(mask)[0]), write_ascii=True)


            # # print("label",label)
            # if label in [4]:
            #     continue
            #     source_points = np.zeros((0, 3))
            #     map_points = np.zeros((0, 3))
            # # print(list(classesColors.keys())[label-1])
            # elif label not in [3]:
            #     sub_pcd = pcd.select_by_index(np.where(mask)[0])
            #     source_points, map_points = pca_downsample(sub_pcd)
            # else: #its ladder
            #     source_points = np.asarray(pcd.select_by_index(np.where(mask)[0]).points)
            #     nr_ladder_points = len(source_points)
            #     map_points = source_points
            # source_colors.append(np.tile(list(classesColors.values())[int(label)-1], (source_points.shape[0], 1)))
            # source_geometries.append(source_points)
            # map_geometries.append(map_points)
            # map_colors.append(np.tile(list(classesColors.values())[int(label)-1], (map_points.shape[0], 1)))
       
        
        

        # pcd_source = o3d.geometry.PointCloud(
        #     points=o3d.utility.Vector3dVector(np.vstack(source_geometries)),
        # )
        # pcd_source.colors = o3d.utility.Vector3dVector(np.vstack(source_colors))

        # pcd_map = o3d.geometry.PointCloud(
        #     points=o3d.utility.Vector3dVector(np.vstack(map_geometries))
        # )
        # pcd_map.colors = o3d.utility.Vector3dVector(np.vstack(map_colors))
        
        for label,nr in nr_instances_labels.items():
            
            category = list(classesColors.keys())[int(label)-1]+"_"+str(nr)
            if category in stats:
                stats[category] += 1
            else:
                stats[category] = 1
        

        if nr_ladder_points<100:
            skipped_Frames += 1
            # colors = np.zeros((len(pcd.points), 3))
            # for i in range(len(final_labels)):
            #     # colors[i] = instanceColors[int(final_instances[i]-1)]
            #     colors[i]= list(classesColors.values())[int(final_labels[i])-1]
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([pcd],window_name=fileName)
            # o3d.visualization.draw_geometries([pcd_source],window_name=fileName+" source",width=1920,height=1080)
            # o3d.visualization.draw_geometries([pcd_map],window_name=fileName+" frame",width=1920,height=1080)
    
        # o3d.io.write_point_cloud(f"data/esbjerg/processed4/{fileName}_source.pcd", pcd_source, write_ascii=True)
        # o3d.io.write_point_cloud(f"data/esbjerg/processed4/{fileName}_map.pcd", pcd_map, write_ascii=True)
        
        
        # print("skipped ",skipped_Frames)
        # print(len(pcd.points))
        # print(len(final_labels))
        
print(stats)
for cat,nr in stats.items():
    print(cat+" :",nr/1701)
        