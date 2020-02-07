# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/global_registration.py

import open3d as o3d
import numpy as np
import copy
import time
from sys import argv
from os.path import expanduser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lib.utils import *

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
    print("source_down.points data type: ",type(source_down.points))
    print("source_down.points data shape: ",len(source_down.points))
    print("source_fpfh data type: ",type(source_fpfh.data))
    print("source_fpfh data shape: ",source_fpfh.data.shape)
    #print(source_fpfh.data)
    plt.imshow(source_fpfh.data[:,0:100], cmap="gray")
    plt.show()
    print("")
    print("target_down.points data type: ",type(target_down.points))
    print("target_down.points data shape: ",len(target_down.points))
    print("target_fpfh data type: ",type(target_fpfh.data))
    print("target_fpfh data shape: ",target_fpfh.data.shape)
    plt.imshow(target_fpfh.data[:,0:100], cmap="hot")
    plt.show()           
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


if __name__ == "__main__":
    method = argv[1]    
    #voxel_size = 0.05  # means 5cm for the dataset -> original example
    voxel_size = 2  # means 2mm for the dataset -> reactor head model scan
    start = time.time()
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(voxel_size)
    print("Data preparation took %.3f sec.\n" % (time.time() - start))
    #print(type(source_down.data))
    #print(source_down.data.shape)
    #print(type(source_fpfh.data))
    #print(source_fpfh.data.shape)
    print("Initial registration starts")
    if method == 'global':
        start = time.time()
        result = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    elif method == 'fast':
        start = time.time()
        result = execute_fast_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    else:
        print('unknown icp method!')
        quit()
    print("Initial registration ended")    
    print(method + " registration took %.3f sec.\n" % (time.time() - start))
    print(result)
    draw_registration_result(source_down, target_down,
                             result.transformation)
    print("Randomization starts") 
    for i in range(10):
        print("Randomization step : ",i)
        #rand_rotation = rand_rotation_matrix_v2(deflection=0.0, randnums=None)
        #rand_rotation = rand_rotation_matrix_v3(deflection=0.0001)
        #rand_rotation = rand_rotation_matrix_v4(3)
        del_x,del_y,del_z = np.random.uniform(size=(3,)) 
        #print(np.random.uniform(size=(3,1)))
        #print(rand_rotation)
        #print(type(rand_rotation))
        #temp_transformation = rotation_only_transformation(rand_rotation)
        temp_transformation = translation_only_transformation(30.0*(del_x-0.5),30.0*(del_y-0.5),50.0*(del_z-0.5))
        np.set_printoptions(precision=3)
        print(temp_transformation)
        print(result.transformation)
  
        #tem = np.dot(result.transformation,temp_transformation)
        #print('translation applied')
        #print(tem)
        tem = np.dot(temp_transformation, result.transformation)
        print('translation applied')
        print(tem)
        #print(np.dot(result.transformation,temp_transformation))
        #print(np.linalg.det(tem))
        #trans_rand = result.transformation + 
        
        temp_result = copy.deepcopy(result)
        temp_result.transformation = tem
        draw_registration_result_downsample(source, target, temp_result.transformation, 3)
        #temp_result.transformation = temp_result.transformation + temp_transformation
        print('refinement starts')
        temp_result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                     voxel_size*1.5, temp_result)
        draw_registration_result_downsample(source, target, temp_result_icp.transformation, 2)

    #source_temp.transform(transformation)
    #start = time.time()
    #result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
    #                                 voxel_size, result)
    #print("Refinement took %.3f sec.\n" % (time.time() - start))
    #print(result_icp)
    #draw_registration_result_downsample(source, target, result_icp.transformation, 2)
