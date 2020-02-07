# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/outlier_removal.py

import open3d as o3d
from os.path import expanduser
import numpy as np

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

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


if __name__ == "__main__":
    home = expanduser("~")
    print("Load a ply point cloud, print it, and render it")
    #pcd = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_2.pcd")
    voxel_size = 1
    pcd_data = 'scan03'
    pcd = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data+'/'+pcd_data+'.ply',format='ply')    
    o3d.visualization.draw_geometries([pcd])

    print("Downsample the point cloud with a voxel size of ",voxel_size)
    #voxel_down_pcd = pcd.voxel_down_sample(voxel_size*2.0)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size)
    o3d.visualization.draw_geometries([voxel_down_pcd])

    #print("Every 5th points are selected")
    #uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
    #o3d.visualization.draw_geometries([uni_down_pcd])
    #remove points near origin
    print("Near-origin points removal")
    distance_criteria = 30 #30mm from the origin (0,0,0)
    near_origin_removed_pcd, ind_nor = remove_near_origin_points(voxel_down_pcd,distance_criteria)
    display_inlier_outlier(voxel_down_pcd, ind_nor)

    print("Statistical oulier removal")
    cl, ind = near_origin_removed_pcd.remove_statistical_outlier(nb_neighbors=30,
                                                         std_ratio=1.0)
    #cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
    #                                                    std_ratio=2.0)    
    
    #cl = selected points (inlier, noise removed)
    #ind = index of selected points
    #print(cl) --> geometry::PointCloud with 25863 points.
    #print(len(ind)) --> 25863
    display_inlier_outlier(near_origin_removed_pcd, ind)
    #display_inlier_outlier(voxel_down_pcd, ind)

    print("Radius oulier removal")
    cl, ind = near_origin_removed_pcd.remove_radius_outlier(nb_points=16, radius=voxel_size*6.0)
    display_inlier_outlier(near_origin_removed_pcd, ind)

    #cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=voxel_size*2.0)
    #display_inlier_outlier(voxel_down_pcd, ind)
