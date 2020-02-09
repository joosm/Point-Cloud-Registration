# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/global_registration.py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import open3d as o3d
import numpy as np
import copy
import time
from sys import argv
from os.path import expanduser

from lib.utils import *

import seaborn as sns
import sklearn.datasets as data
import hdbscan

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def draw_registration_result_downsample(source, target, transformation, voxel_size):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    source_temp = source_temp.voxel_down_sample(voxel_size=4.0)
    target_temp = target_temp.voxel_down_sample(voxel_size=4.0)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print("downsample size:", len(pcd_down.points))

    radius_normal = voxel_size * 2
    #radius_normal = voxel_size * 3
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #radius_feature = voxel_size * 2
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def preprocess_point_cloud_wo_downsampling(pcd, voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    #pcd_down = pcd.voxel_down_sample(voxel_size)
    #print("downsample size:", len(pcd_down.points))

    radius_normal = voxel_size * 5
    #radius_normal = voxel_size * 3
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    #radius_feature = voxel_size * 5
    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh


def prepare_dataset(voxel_size):
    home = expanduser("~")
    print(":: Load two point clouds and disturb initial pose.")
    #source = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    #target = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
    source_data = 'scan03'
    target_data = 'scan04'
    source = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+source_data+'/'+source_data+'.ply',format='ply')
    target = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+target_data+'/'+target_data+'.ply',format='ply') 
    trans_init =  np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))
    #draw_registration_result_downsample(source, target, np.identity(4), 3)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    #source_feature_data = source_fpfh.data[:,0:100]
    #target_feature_data = target_fpfh.data[:,0:100]
    print("source_down.points data type: ",type(source_down.points))
    print("source_down.points data shape: ",len(source_down.points))
    print("source_fpfh data type: ",type(source_fpfh.data))
    print("source_fpfh data shape: ",source_fpfh.data.shape)
    #print(source_fpfh.data)

    plt.imshow(source_fpfh.data[:,0:10], cmap="gray")
    print("test imshow")
    plt.show()
    print("")
    print("target_down.points data type: ",type(target_down.points))
    print("target_down.points data shape: ",len(target_down.points))
    print("target_fpfh data type: ",type(target_fpfh.data))
    print("target_fpfh data shape: ",target_fpfh.data.shape)
    #plt.imshow(target_fpfh.data[:,0:100], cmap="hot")
    #plt.show()           
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_of_initial_step):
    #distance_threshold = voxel_size * 0.4
    distance_threshold = voxel_size * 1.0
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_of_initial_step.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result


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

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

if __name__ == "__main__":

    home = expanduser("~")
    print("Load a ply point cloud, print it, and render it")
    #pcd = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_2.pcd")
    voxel_size = 2

    pcd_data = 'scan03'
    pcd = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data+'/'+pcd_data+'.ply',format='ply')    
    #o3d.visualization.draw_geometries([pcd])
    print("Original data size: ", len(pcd.points))

    print("Downsample the point cloud with a voxel size of ",voxel_size)
    #voxel_down_pcd = pcd.voxel_down_sample(voxel_size*2.0)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size)
    print("Down-sampled data size: ", len(voxel_down_pcd.points))
    #o3d.visualization.draw_geometries([voxel_down_pcd])

    #print("Every 5th points are selected")
    #uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
    #o3d.visualization.draw_geometries([uni_down_pcd])

    #remove points near origin
    print("Near-origin points removal")
    distance_criteria = 30 #30mm from the origin (0,0,0)
    near_origin_removed_pcd, ind_nor = remove_near_origin_points(voxel_down_pcd,distance_criteria)
    #display_inlier_outlier(voxel_down_pcd, ind_nor)

    print("Statistical oulier removal")
    statistical_outlier_removed_pcd, ind_sta = near_origin_removed_pcd.remove_statistical_outlier(nb_neighbors=30,
                                                         std_ratio=1.0)
    #cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
    #                                                    std_ratio=2.0)    
    
    #cl = selected points (inlier, noise removed)
    #ind = index of selected points
    #print(cl) --> geometry::PointCloud with 25863 points.
    #print(len(ind)) --> 25863
    #display_inlier_outlier(near_origin_removed_pcd, ind)
    #display_inlier_outlier(voxel_down_pcd, ind)

    print("Radius oulier removal")
    radius_outlier_removed_pcd, ind_rad = statistical_outlier_removed_pcd.remove_radius_outlier(nb_points=16, radius=voxel_size*6.0)
    print("Outlier removed data size: ", len(radius_outlier_removed_pcd.points))
    #display_inlier_outlier(near_origin_removed_pcd, ind)
    
    o3d.visualization.draw_geometries([radius_outlier_removed_pcd])
    
    #cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=voxel_size*2.0)
    #display_inlier_outlier(voxel_down_pcd, ind)

    #voxel_size = 0.05  # means 5cm for the dataset -> original example
    #voxel_size = 2  # means 2mm for the dataset -> reactor head model scan
    #start = time.time()
    #pcd_data_down, pcd_fpfh = preprocess_point_cloud(radius_outlier_removed_pcd, voxel_size)
    pcd_data_down, pcd_fpfh = preprocess_point_cloud_wo_downsampling(radius_outlier_removed_pcd, voxel_size)
    print("Feature-sampled data size: ", len(pcd_data_down.points))
    #print("Data preparation took %.3f sec.\n" % (time.time() - start))
    plt.figure(num=1)
    #plt.imshow(pcd_fpfh.data[:,0:100].T, cmap="gray")
    plt.imshow(pcd_fpfh.data[:,0:100], cmap="gray")
    print("test imshow")
    
    #source, target, source_down, target_down, source_fpfh, target_fpfh = \
    #        prepare_dataset(voxel_size)
    #clusterer = hdbscan.HDBSCAN(min_cluster_size=250, gen_min_span_tree=True)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=250, prediction_data=True)
    clusterer.fit(pcd_fpfh.data.T)

    #plt.figure(num=2)
    #clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=10, edge_linewidth=2)
    #plt.show()

    #plt.figure(num=2)
    #clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    #plt.show()

    plt.figure(num=2)
    #clusterer.condensed_tree_.plot()
    clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
    plt.show()
    

