# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/interactive_visualization.py

import numpy as np
import copy
import open3d as o3d
from os.path import expanduser


def rotation_only_transformation(rotation_matrix):
    trans =  np.asarray([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    trans[0:3,0:3] = rotation_matrix
    return trans

def translation_only_transformation(x,y,z):
    trans =  np.asarray([[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y],
                        [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]])
    #trans[3,0:3] = translation_vector
    return trans   

def remove_near_origin_points(pcd,distance_criteria):
    #pcd = o3d.io.read_point_cloud(point_cloud,format='ply') 
    ind = []
    #cl = o3d.geometry.PointCloud()

    for i in range(len(pcd.points)):
        if np.linalg.norm(pcd.points[i],ord=None) > distance_criteria:
            #xyz = np.asarray(pcd.points[i])
            #print(i,' ',xyz)
            #print(i,' ',np.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2), np.linalg.norm(pcd.points[i],ord=None))
            #print(np.sqrt(pcd.points[i,0]**2 + pcd.points[i,1]**2 + pcd.points[i,2]**2), np.linalg.norm(pcd.points[i],ord=None) )
            #print(i)
            ind.append(i)
    cl = pcd.select_down_sample(ind)

    return cl, ind

def downsample_remove_outlier(pcd, distance_criteria, voxel_size):
    #print("Near-origin points removal")
    #near_origin_removed_pcd, ind_nor = remove_near_origin_points(pcd,distance_criteria)
    down_pcd = pcd.voxel_down_sample(voxel_size)
    near_origin_removed_pcd = down_pcd
    print("Statistical oulier removal")
    cl, ind = near_origin_removed_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    print("Radius oulier removal")
    #cl, ind = near_origin_removed_pcd.remove_radius_outlier(nb_points=16, radius=voxel_size*6.0)
    cl, ind = near_origin_removed_pcd.remove_radius_outlier(nb_points=20, radius=voxel_size*3.0)
    return cl, ind

def demo_crop_geometry():
    print("Demo for manual geometry cropping")
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")
    pcd = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    o3d.visualization.draw_geometries_with_editing([pcd])


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.0, 0])#([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929]) #
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def demo_manual_registration():

    print("Demo for manual ICP")
    #source = o3d.io.read_point_cloud("../Open3D/examples/TestData/ICP/cloud_bin_0.pcd")
    #target = o3d.io.read_point_cloud("../Open3D/examples/TestData/ICP/cloud_bin_2.pcd")
    home = expanduser("~")
    distance_criteria = 40
    voxel_size = 1
    pcd_data = 'scan08'
    temp_source = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data+'/'+pcd_data+'.ply',format='ply')
    source, ind = downsample_remove_outlier(temp_source, distance_criteria,voxel_size)  
    pcd_data = 'scan09'    
    temp_target = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data+'/'+pcd_data+'.ply',format='ply')
    target, ind = downsample_remove_outlier(temp_target, distance_criteria,voxel_size)      
    #source = o3d.io.read_point_cloud("../Open3D/examples/TestData/ICP/cloud_bin_0.pcd")
    #target = o3d.io.read_point_cloud("../Open3D/examples/TestData/ICP/cloud_bin_2.pcd")

    print("Visualization of two point clouds before manual alignment")
    #draw_registration_result(source, target, np.identity(4))
    draw_registration_result(source, target, translation_only_transformation(400,0,0))

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 1.5 #0.02  # 3cm distance threshold
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
    draw_registration_result(source, target, reg_p2p.transformation)
    print("")


if __name__ == "__main__":
    #demo_crop_geometry()
    demo_manual_registration()
