# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/interactive_visualization.py

import numpy as np
import copy
import open3d as o3d
from os.path import expanduser
from os.path import isfile
import color_dictionary as cd

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


def pairwise_manual_registration(source_pcd_name, target_pcd_name, distance_criteria, voxel_size, threshold):
    home = expanduser("~")
    #distance_criteria = 40
    #voxel_size = 1
    #threshold = 1.5 #0.02  # 3cm distance threshold   

    source_pcd_data = source_pcd_name
    temp_source = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+source_pcd_data+'/'+source_pcd_data+'.ply',format='ply')
    source, ind = downsample_remove_outlier(temp_source, distance_criteria,voxel_size)  

    target_pcd_data = target_pcd_name    
    temp_target = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+target_pcd_data+'/'+target_pcd_data+'.ply',format='ply')
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
    trans_init = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")

    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())

    draw_registration_result(source, target, reg_p2p.transformation)

    print("")
    np.savez('./pairwise_registration/'+source_pcd_data+'_'+target_pcd_data+'_transformations',initial_transformation = trans_init, refined_transformation =  reg_p2p.transformation )
    if not isfile('./pairwise_registration/'+source_pcd_data+'_clean_downsampled.ply'):
        o3d.io.write_point_cloud('./pairwise_registration/'+source_pcd_data+'_clean_downsampled.ply', source)
    if not isfile('./pairwise_registration/'+target_pcd_data+'_clean_downsampled.ply'):    
        o3d.io.write_point_cloud('./pairwise_registration/'+target_pcd_data+'_clean_downsampled.ply', source)


def demo_manual_registration():

    print("Demo for manual ICP")
    #source = o3d.io.read_point_cloud("../Open3D/examples/TestData/ICP/cloud_bin_0.pcd")
    #target = o3d.io.read_point_cloud("../Open3D/examples/TestData/ICP/cloud_bin_2.pcd")
    home = expanduser("~")
    distance_criteria = 40
    voxel_size = 1
    source_pcd_data = 'scan08'
    temp_source = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+source_pcd_data+'/'+source_pcd_data+'.ply',format='ply')
    source, ind = downsample_remove_outlier(temp_source, distance_criteria,voxel_size)  

    target_pcd_data = 'scan09'    
    temp_target = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+target_pcd_data+'/'+target_pcd_data+'.ply',format='ply')
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
    #np.savez(source_pcd_data+'_'+target_pcd_data+'_transformations',initial_transformation = trans_init, refined_transformation =  reg_p2p.transformation )
    #o3d.io.write_point_cloud(source_pcd_data+'_clean_downsampled.ply', source)
    #o3d.io.write_point_cloud(target_pcd_data+'_clean_downsampled.ply', source)
    #np.savez_compressed('transformations',initial_transformation = trans_init, refined_transformation =  reg_p2p.transformation )

def load_pairwise_registration(source_name, target_name, source_color, target_color):
    home = expanduser("~")
    source_pcd_data = source_name
    #temp_source = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+source_pcd_data+'/'+source_pcd_data+'.ply',format='ply')
    temp_source = o3d.io.read_point_cloud(home+'/Workplace/Point-Cloud-Registration/pairwise_registration/'+source_pcd_data+'_clean_downsampled.ply',format='ply')
    temp_source.paint_uniform_color(source_color)

    target_pcd_data = target_name    
    #temp_target = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+target_pcd_data+'/'+target_pcd_data+'.ply',format='ply')
    temp_target = o3d.io.read_point_cloud(home+'/Workplace/Point-Cloud-Registration/pairwise_registration/'+target_pcd_data+'_clean_downsampled.ply',format='ply')
    temp_target.paint_uniform_color(target_color)
    transformation = np.load('./pairwise_registration/'+source_pcd_data+'_'+target_pcd_data+'_transformations.npz')

    temp_source.transform(transformation['refined_transformation'])

    return temp_source, temp_target

def load_pairwise_registration_simple(source_name, target_name, source_color, target_color):
    home = expanduser("~")
    source_pcd_data = source_name
    temp_source = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+source_pcd_data+'/'+source_pcd_data+'.ply',format='ply')
    #temp_source = o3d.io.read_point_cloud(home+'/Workplace/Point-Cloud-Registration/pairwise_registration/'+source_pcd_data+'_clean_downsampled.ply',format='ply')
    temp_source.paint_uniform_color(source_color)

    target_pcd_data = target_name    
    temp_target = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+target_pcd_data+'/'+target_pcd_data+'.ply',format='ply')
    #temp_target = o3d.io.read_point_cloud(home+'/Workplace/Point-Cloud-Registration/pairwise_registration/'+target_pcd_data+'_clean_downsampled.ply',format='ply')
    temp_target.paint_uniform_color(target_color)
    
    transformation = np.load('./pairwise_registration/'+source_pcd_data+'_'+target_pcd_data+'_transformations.npz')

    #temp_source.transform(transformation['refined_transformation'])

    return temp_source, temp_target, transformation['refined_transformation']    

def merge_incremental_registration(source_pcd, target_pcd, source_to_target_transformation):
    #temp_pcd = []
    #temp_pcd.append(target_pcd)
    #print(source_to_target_transformation)
    
    #temp_pcd.append(source_pcd.transform(source_to_target_transformation))
    temp_pcd = source_pcd.transform(source_to_target_transformation)
    #return temp_pcd
    return temp_pcd + target_pcd

def merge_pcd(source_pcd, target_pcd):
    #temp_pcd = []
    #temp_pcd.append(target_pcd)
    #temp_pcd.append(source_pcd)

    #return temp_pcd
    return source_pcd + target_pcd

def apply_transformations(pcd, transformations):
    temp_pcd = copy.deepcopy(pcd)
    for tf in transformations:
        temp_pcd.transform(tf)
    return temp_pcd

if __name__ == "__main__":
    #demo_crop_geometry()
    #demo_manual_registration()


    #home = expanduser("~")
    #source_pcd_data = 'scan01'
    ##temp_source = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+source_pcd_data+'/'+source_pcd_data+'.ply',format='ply')
    #temp_source = o3d.io.read_point_cloud(home+'/Workplace/Point-Cloud-Registration/pairwise_registration/'+source_pcd_data+'_clean_downsampled.ply',format='ply')
    #temp_source.paint_uniform_color(cd.ALICEBLUE.as_ndarray())

    #target_pcd_data = 'scan02'    
    ##temp_target = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+target_pcd_data+'/'+target_pcd_data+'.ply',format='ply')
    #temp_target = o3d.io.read_point_cloud(home+'/Workplace/Point-Cloud-Registration/pairwise_registration/'+target_pcd_data+'_clean_downsampled.ply',format='ply')
    #temp_target.paint_uniform_color(cd.RED1.as_ndarray())
    #transformation = np.load('./pairwise_registration/'+source_pcd_data+'_'+target_pcd_data+'_transformations.npz')

    #temp_source.transform(transformation['refined_transformation'])


    #registered_scan01, scan02 = load_pairwise_registration('scan01','scan02',cd.RED.as_ndarray(),cd.LIME.as_ndarray())
    #scan01, scan02, tf_scan01_to_scan02 = load_pairwise_registration_simple('scan01','scan02',cd.RED1.as_ndarray(),cd.LIMEGREEN.as_ndarray())
    #scan01_scan02 = merge_incremental_registration(scan01, scan02, tf_scan01_to_scan02)

    #scan02, scan03, tf_scan02_to_scan03 = load_pairwise_registration_simple('scan02','scan03',cd.LIMEGREEN.as_ndarray(),cd.BLUE.as_ndarray())
    #scan01_scan02_scan03 = merge_incremental_registration(scan01_scan02, scan03, tf_scan02_to_scan03)
    
    #o3d.visualization.draw_geometries([scan01,scan02])
    #draw_registration_result(scan01, scan02, tf_scan01_to_scan02)
    #o3d.visualization.draw_geometries([scan01_scan02])
    #o3d.visualization.draw_geometries([scan01_scan02_scan03])
    #print(cd.ALICEBLUE.as_ndarray())
    #pairwise_manual_registration(source_pcd_name='scan01', target_pcd_name='scan02', distance_criteria=40, voxel_size=1, threshold=1.5)
    #pairwise_manual_registration(source_pcd_name='scan02', target_pcd_name='scan03', distance_criteria=40, voxel_size=1, threshold=1.5)
    #pairwise_manual_registration(source_pcd_name='scan03', target_pcd_name='scan04', distance_criteria=40, voxel_size=1, threshold=1.5) 
    #pairwise_manual_registration(source_pcd_name='scan03', target_pcd_name='scan04', distance_criteria=40, voxel_size=1, threshold=1.5)  
    #pairwise_manual_registration(source_pcd_name='scan04', target_pcd_name='scan05', distance_criteria=40, voxel_size=1, threshold=1.5) 
    #pairwise_manual_registration(source_pcd_name='scan05', target_pcd_name='scan06', distance_criteria=40, voxel_size=1, threshold=1.5) 
    #pairwise_manual_registration(source_pcd_name='scan06', target_pcd_name='scan07', distance_criteria=40, voxel_size=1, threshold=1.0) 
    #pairwise_manual_registration(source_pcd_name='scan07', target_pcd_name='scan08', distance_criteria=40, voxel_size=1, threshold=1.5) 
    #pairwise_manual_registration(source_pcd_name='scan08', target_pcd_name='scan09', distance_criteria=40, voxel_size=1, threshold=1.5) 
    #pairwise_manual_registration(source_pcd_name='scan09', target_pcd_name='scan10', distance_criteria=40, voxel_size=1, threshold=1.5)
    #pairwise_manual_registration(source_pcd_name='scan10', target_pcd_name='scan11', distance_criteria=40, voxel_size=1, threshold=1.5)
    #pairwise_manual_registration(source_pcd_name='scan11', target_pcd_name='scan12', distance_criteria=40, voxel_size=1, threshold=1.5)
    #pairwise_manual_registration(source_pcd_name='scan12', target_pcd_name='scan13', distance_criteria=40, voxel_size=1, threshold=1.5)
    #pairwise_manual_registration(source_pcd_name='scan13', target_pcd_name='scan14', distance_criteria=40, voxel_size=1, threshold=1.5)
    #pairwise_manual_registration(source_pcd_name='scan14', target_pcd_name='scan15', distance_criteria=40, voxel_size=1, threshold=1.5)
    #pairwise_manual_registration(source_pcd_name='scan15', target_pcd_name='scan16', distance_criteria=40, voxel_size=1, threshold=1.5)
    home = expanduser("~")
    my_dict = {}

    pcd_data_name = 'scan01'
    target_pcd_data_name = 'scan02'
    temp_pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')
    #o3d.visualization.draw_geometries([temp_pcd_data])
    pcd_name = pcd_data_name + '_pcd'
    my_dict[pcd_name], ind = downsample_remove_outlier(temp_pcd_data, distance_criteria = 40, voxel_size = 1)
    my_dict[pcd_name].paint_uniform_color(cd.RED1.as_ndarray())
    temp_tf = np.load('./pairwise_registration/'+pcd_data_name+'_'+target_pcd_data_name+'_transformations.npz')
    tf_01_02 = temp_tf['refined_transformation']
    #o3d.visualization.draw_geometries([pcd_name])

    pcd_data_name = 'scan02'
    target_pcd_data_name = 'scan03'    
    temp_pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')
    #o3d.visualization.draw_geometries([temp_pcd_data])
    pcd_name = pcd_data_name + '_pcd'
    my_dict[pcd_name], ind = downsample_remove_outlier(temp_pcd_data, distance_criteria = 40, voxel_size = 1)
    my_dict[pcd_name].paint_uniform_color(cd.LIMEGREEN.as_ndarray())
    temp_tf = np.load('./pairwise_registration/'+pcd_data_name+'_'+target_pcd_data_name+'_transformations.npz')
    tf_02_03 = temp_tf['refined_transformation']

    pcd_data_name = 'scan03'
    target_pcd_data_name = 'scan04'
    temp_pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')
    #o3d.visualization.draw_geometries([temp_pcd_data])
    pcd_name = pcd_data_name + '_pcd'
    my_dict[pcd_name], ind = downsample_remove_outlier(temp_pcd_data, distance_criteria = 40, voxel_size = 1)
    #= temp
    my_dict[pcd_name].paint_uniform_color(cd.BLUE.as_ndarray())
    temp_tf = np.load('./pairwise_registration/'+pcd_data_name+'_'+target_pcd_data_name+'_transformations.npz')
    tf_03_04 = temp_tf['refined_transformation']


    pcd_data_name = 'scan04'
    target_pcd_data_name = 'scan05'
    temp_pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')
    #o3d.visualization.draw_geometries([temp_pcd_data])
    pcd_name = pcd_data_name + '_pcd'
    my_dict[pcd_name], ind = downsample_remove_outlier(temp_pcd_data, distance_criteria = 40, voxel_size = 1)
    #= temp
    my_dict[pcd_name].paint_uniform_color(cd.YELLOW1.as_ndarray())
    temp_tf = np.load('./pairwise_registration/'+pcd_data_name+'_'+target_pcd_data_name+'_transformations.npz')
    tf_04_05 = temp_tf['refined_transformation']

    pcd_data_name = 'scan05'
    target_pcd_data_name = 'scan06'
    temp_pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')
    #o3d.visualization.draw_geometries([temp_pcd_data])
    pcd_name = pcd_data_name + '_pcd'
    my_dict[pcd_name], ind = downsample_remove_outlier(temp_pcd_data, distance_criteria = 40, voxel_size = 1)
    #= temp
    my_dict[pcd_name].paint_uniform_color(cd.CYAN2.as_ndarray())
    temp_tf = np.load('./pairwise_registration/'+pcd_data_name+'_'+target_pcd_data_name+'_transformations.npz')
    tf_05_06 = temp_tf['refined_transformation']

    pcd_data_name = 'scan06'
    target_pcd_data_name = 'scan07'
    temp_pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')
    #o3d.visualization.draw_geometries([temp_pcd_data])
    pcd_name = pcd_data_name + '_pcd'
    my_dict[pcd_name], ind = downsample_remove_outlier(temp_pcd_data, distance_criteria = 40, voxel_size = 1)
    #= temp
    my_dict[pcd_name].paint_uniform_color(cd.MAGENTA.as_ndarray())
    temp_tf = np.load('./pairwise_registration/'+pcd_data_name+'_'+target_pcd_data_name+'_transformations.npz')
    tf_06_07 = temp_tf['refined_transformation']

    pcd_data_name = 'scan07'
    target_pcd_data_name = 'scan08'
    temp_pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')
    #o3d.visualization.draw_geometries([temp_pcd_data])
    pcd_name = pcd_data_name + '_pcd'
    my_dict[pcd_name], ind = downsample_remove_outlier(temp_pcd_data, distance_criteria = 40, voxel_size = 1)
    #= temp
    my_dict[pcd_name].paint_uniform_color(cd.SILVER.as_ndarray())
    temp_tf = np.load('./pairwise_registration/'+pcd_data_name+'_'+target_pcd_data_name+'_transformations.npz')
    tf_07_08 = temp_tf['refined_transformation']

    pcd_data_name = 'scan08'
    target_pcd_data_name = 'scan09'
    temp_pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')
    #o3d.visualization.draw_geometries([temp_pcd_data])
    pcd_name = pcd_data_name + '_pcd'
    my_dict[pcd_name], ind = downsample_remove_outlier(temp_pcd_data, distance_criteria = 40, voxel_size = 1)
    #= temp
    my_dict[pcd_name].paint_uniform_color(cd.MAROON.as_ndarray())
    temp_tf = np.load('./pairwise_registration/'+pcd_data_name+'_'+target_pcd_data_name+'_transformations.npz')
    tf_08_09 = temp_tf['refined_transformation'] 

    pcd_data_name = 'scan09'
    target_pcd_data_name = 'scan10'
    temp_pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')
    #o3d.visualization.draw_geometries([temp_pcd_data])
    pcd_name = pcd_data_name + '_pcd'
    my_dict[pcd_name], ind = downsample_remove_outlier(temp_pcd_data, distance_criteria = 40, voxel_size = 1)
    #= temp
    my_dict[pcd_name].paint_uniform_color(cd.OLIVE.as_ndarray())
    temp_tf = np.load('./pairwise_registration/'+pcd_data_name+'_'+target_pcd_data_name+'_transformations.npz')
    tf_09_10 = temp_tf['refined_transformation'] 

    pcd_data_name = 'scan10'
    target_pcd_data_name = 'scan11'
    temp_pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')
    #o3d.visualization.draw_geometries([temp_pcd_data])
    pcd_name = pcd_data_name + '_pcd'
    my_dict[pcd_name], ind = downsample_remove_outlier(temp_pcd_data, distance_criteria = 40, voxel_size = 1)
    #= temp
    my_dict[pcd_name].paint_uniform_color(cd.TEAL.as_ndarray())
    temp_tf = np.load('./pairwise_registration/'+pcd_data_name+'_'+target_pcd_data_name+'_transformations.npz')
    tf_10_11 = temp_tf['refined_transformation']    


    pcd_data_name = 'scan11'
    target_pcd_data_name = 'scan12'
    temp_pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')
    #o3d.visualization.draw_geometries([temp_pcd_data])
    pcd_name = pcd_data_name + '_pcd'
    my_dict[pcd_name], ind = downsample_remove_outlier(temp_pcd_data, distance_criteria = 40, voxel_size = 1)
    #= temp
    my_dict[pcd_name].paint_uniform_color(cd.PURPLE.as_ndarray())
    temp_tf = np.load('./pairwise_registration/'+pcd_data_name+'_'+target_pcd_data_name+'_transformations.npz')
    tf_11_12 = temp_tf['refined_transformation']

    pcd_data_name = 'scan12'
    target_pcd_data_name = 'scan13'
    temp_pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')
    #o3d.visualization.draw_geometries([temp_pcd_data])
    pcd_name = pcd_data_name + '_pcd'
    my_dict[pcd_name], ind = downsample_remove_outlier(temp_pcd_data, distance_criteria = 40, voxel_size = 1)
    #= temp
    my_dict[pcd_name].paint_uniform_color(cd.NAVY.as_ndarray())
    temp_tf = np.load('./pairwise_registration/'+pcd_data_name+'_'+target_pcd_data_name+'_transformations.npz')
    tf_12_13 = temp_tf['refined_transformation']  

    pcd_data_name = 'scan13'
    target_pcd_data_name = 'scan14'
    temp_pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')
    #o3d.visualization.draw_geometries([temp_pcd_data])
    pcd_name = pcd_data_name + '_pcd'
    my_dict[pcd_name], ind = downsample_remove_outlier(temp_pcd_data, distance_criteria = 40, voxel_size = 1)
    #= temp
    my_dict[pcd_name].paint_uniform_color(cd.LAVENDER.as_ndarray())
    temp_tf = np.load('./pairwise_registration/'+pcd_data_name+'_'+target_pcd_data_name+'_transformations.npz')
    tf_13_14 = temp_tf['refined_transformation']  

    pcd_data_name = 'scan14'
    target_pcd_data_name = 'scan15'
    temp_pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')
    #o3d.visualization.draw_geometries([temp_pcd_data])
    pcd_name = pcd_data_name + '_pcd'
    my_dict[pcd_name], ind = downsample_remove_outlier(temp_pcd_data, distance_criteria = 40, voxel_size = 1)
    #= temp
    my_dict[pcd_name].paint_uniform_color(cd.LIGHTSALMON1.as_ndarray())
    temp_tf = np.load('./pairwise_registration/'+pcd_data_name+'_'+target_pcd_data_name+'_transformations.npz')
    tf_14_15 = temp_tf['refined_transformation']  

    pcd_data_name = 'scan15'
    target_pcd_data_name = 'scan16'
    temp_pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')
    #o3d.visualization.draw_geometries([temp_pcd_data])
    pcd_name = pcd_data_name + '_pcd'
    my_dict[pcd_name], ind = downsample_remove_outlier(temp_pcd_data, distance_criteria = 40, voxel_size = 1)
    #= temp
    my_dict[pcd_name].paint_uniform_color(cd.ORANGE.as_ndarray())
    temp_tf = np.load('./pairwise_registration/'+pcd_data_name+'_'+target_pcd_data_name+'_transformations.npz')
    tf_15_16 = temp_tf['refined_transformation'] 


    print("phase 1 ends")


    #o3d.visualization.draw_geometries([my_dict['scan02_pcd']])

    #o3d.visualization.draw_geometries([my_dict['scan02_pcd'], my_dict['scan01_pcd'].transform(tf_01_02)])

    
    #o3d.visualization.draw_geometries([my_dict['scan03_pcd'], my_dict['scan02_pcd'].transform(tf_02_03)
    #    ,apply_transformations(my_dict['scan01_pcd'],[tf_01_02,tf_02_03])])

    
    #o3d.visualization.draw_geometries([my_dict['scan05_pcd'], my_dict['scan04_pcd'].transform(tf_04_05)
    #    ,apply_transformations(my_dict['scan03_pcd'],[tf_03_04,tf_04_05])
    #    ,apply_transformations(my_dict['scan02_pcd'],[tf_02_03,tf_03_04,tf_04_05])
    #    ,apply_transformations(my_dict['scan01_pcd'],[tf_01_02,tf_02_03,tf_03_04,tf_04_05])])
    
    
    #o3d.visualization.draw_geometries([my_dict['scan06_pcd'], my_dict['scan05_pcd'].transform(tf_05_06)
    #    ,apply_transformations(my_dict['scan04_pcd'],[tf_04_05,tf_05_06])        
    #    ,apply_transformations(my_dict['scan03_pcd'],[tf_03_04,tf_04_05,tf_05_06])
    #    ,apply_transformations(my_dict['scan02_pcd'],[tf_02_03,tf_03_04,tf_04_05,tf_05_06])
    #    ,apply_transformations(my_dict['scan01_pcd'],[tf_01_02,tf_02_03,tf_03_04,tf_04_05,tf_05_06])])    

    '''
    o3d.visualization.draw_geometries([my_dict['scan07_pcd'], my_dict['scan06_pcd'].transform(tf_06_07)
        ,apply_transformations(my_dict['scan05_pcd'],[tf_05_06,tf_06_07])
        ,apply_transformations(my_dict['scan04_pcd'],[tf_04_05,tf_05_06,tf_06_07])
        ,apply_transformations(my_dict['scan03_pcd'],[tf_03_04,tf_04_05,tf_05_06,tf_06_07])
        ,apply_transformations(my_dict['scan02_pcd'],[tf_02_03,tf_03_04,tf_04_05,tf_05_06,tf_06_07])
        ,apply_transformations(my_dict['scan01_pcd'],[tf_01_02,tf_02_03,tf_03_04,tf_04_05,tf_05_06,tf_06_07])])
    '''
    
    o3d.visualization.draw_geometries([my_dict['scan10_pcd'], my_dict['scan09_pcd'].transform(tf_09_10)
        ,apply_transformations(my_dict['scan08_pcd'],[tf_08_09,tf_09_10])
        ,apply_transformations(my_dict['scan07_pcd'],[tf_07_08,tf_08_09,tf_09_10])
        ,apply_transformations(my_dict['scan06_pcd'],[tf_06_07,tf_07_08,tf_08_09,tf_09_10])
        ,apply_transformations(my_dict['scan05_pcd'],[tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10])
        ,apply_transformations(my_dict['scan04_pcd'],[tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10])
        ,apply_transformations(my_dict['scan03_pcd'],[tf_03_04,tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10])
        ,apply_transformations(my_dict['scan02_pcd'],[tf_02_03,tf_03_04,tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10])
        ,apply_transformations(my_dict['scan01_pcd'],[tf_01_02,tf_02_03,tf_03_04,tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10])])
    
    '''
    o3d.visualization.draw_geometries([my_dict['scan11_pcd'], my_dict['scan10_pcd'].transform(tf_10_11)
        ,apply_transformations(my_dict['scan09_pcd'],[tf_09_10,tf_10_11])
        ,apply_transformations(my_dict['scan08_pcd'],[tf_08_09,tf_09_10,tf_10_11])
        ,apply_transformations(my_dict['scan07_pcd'],[tf_07_08,tf_08_09,tf_09_10,tf_10_11])
        ,apply_transformations(my_dict['scan06_pcd'],[tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11])
        ,apply_transformations(my_dict['scan05_pcd'],[tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11])
        ,apply_transformations(my_dict['scan04_pcd'],[tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11])
        ,apply_transformations(my_dict['scan03_pcd'],[tf_03_04,tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11])
        ,apply_transformations(my_dict['scan02_pcd'],[tf_02_03,tf_03_04,tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11])
        ,apply_transformations(my_dict['scan01_pcd'],[tf_01_02,tf_02_03,tf_03_04,tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11])])

        
    o3d.visualization.draw_geometries([my_dict['scan12_pcd'], my_dict['scan11_pcd'].transform(tf_11_12)
        ,apply_transformations(my_dict['scan10_pcd'],[tf_10_11,tf_11_12])        
        ,apply_transformations(my_dict['scan09_pcd'],[tf_09_10,tf_10_11,tf_11_12])
        ,apply_transformations(my_dict['scan08_pcd'],[tf_08_09,tf_09_10,tf_10_11,tf_11_12])
        ,apply_transformations(my_dict['scan07_pcd'],[tf_07_08,tf_08_09,tf_09_10,tf_10_11,tf_11_12])
        ,apply_transformations(my_dict['scan06_pcd'],[tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11,tf_11_12])
        ,apply_transformations(my_dict['scan05_pcd'],[tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11,tf_11_12])
        ,apply_transformations(my_dict['scan04_pcd'],[tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11,tf_11_12])
        ,apply_transformations(my_dict['scan03_pcd'],[tf_03_04,tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11,tf_11_12])
        ,apply_transformations(my_dict['scan02_pcd'],[tf_02_03,tf_03_04,tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11,tf_11_12])
        ,apply_transformations(my_dict['scan01_pcd'],[tf_01_02,tf_02_03,tf_03_04,tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11,tf_11_12])])  
    
    
    #o3d.visualization.draw_geometries([scan03_pcd, scan02_pcd.transformation(tf_02_03), apply_transformations(scan01_pcd,[tf_01_02,tf_02_03])])

    o3d.visualization.draw_geometries([my_dict['scan15_pcd'],my_dict['scan14_pcd'].transform(tf_14_15)
        ,apply_transformations(my_dict['scan13_pcd'],[tf_13_14,tf_14_15])        
        ,apply_transformations(my_dict['scan12_pcd'],[tf_12_13,tf_13_14,tf_14_15])
        ,apply_transformations(my_dict['scan11_pcd'],[tf_11_12,tf_12_13,tf_13_14,tf_14_15])
        ,apply_transformations(my_dict['scan10_pcd'],[tf_10_11,tf_11_12,tf_12_13,tf_13_14,tf_14_15])
        ,apply_transformations(my_dict['scan09_pcd'],[tf_09_10,tf_10_11,tf_11_12,tf_12_13,tf_13_14,tf_14_15])
        ,apply_transformations(my_dict['scan08_pcd'],[tf_08_09,tf_09_10,tf_10_11,tf_11_12,tf_12_13,tf_13_14,tf_14_15])
        ,apply_transformations(my_dict['scan07_pcd'],[tf_07_08,tf_08_09,tf_09_10,tf_10_11,tf_11_12,tf_12_13,tf_13_14,tf_14_15])
        ,apply_transformations(my_dict['scan06_pcd'],[tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11,tf_11_12,tf_12_13,tf_13_14,tf_14_15])
        ,apply_transformations(my_dict['scan05_pcd'],[tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11,tf_11_12,tf_12_13,tf_13_14,tf_14_15])
        ,apply_transformations(my_dict['scan04_pcd'],[tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11,tf_11_12,tf_12_13,tf_13_14,tf_14_15]) 
        ,apply_transformations(my_dict['scan03_pcd'],[tf_03_04,tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11,tf_11_12,tf_12_13,tf_13_14,tf_14_15]) 
        ,apply_transformations(my_dict['scan02_pcd'],[tf_02_03,tf_03_04,tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11,tf_11_12,tf_12_13,tf_13_14,tf_14_15]) 
        ,apply_transformations(my_dict['scan01_pcd'],[tf_01_02,tf_02_03,tf_03_04,tf_04_05,tf_05_06,tf_06_07,tf_07_08,tf_08_09,tf_09_10,tf_10_11,tf_11_12,tf_12_13,tf_13_14,tf_14_15])]) 
    '''  

    '''
    pcd_name.paint_uniform_color(cd.YELLOW1.as_ndarray())
    pcd_name.paint_uniform_color(cd.CYAN2.as_ndarray())
    pcd_name.paint_uniform_color(cd.MAGENTA.as_ndarray())
    pcd_name.paint_uniform_color(cd.SILVER.as_ndarray())
    pcd_name.paint_uniform_color(cd.GRAY.as_ndarray())
    pcd_name.paint_uniform_color(cd.MAROON.as_ndarray())
    pcd_name.paint_uniform_color(cd.OLIVE.as_ndarray())
    pcd_name.paint_uniform_color(cd.GREEN.as_ndarray())
    pcd_name.paint_uniform_color(cd.PURPLE.as_ndarray())
    pcd_name.paint_uniform_color(cd.TEAL.as_ndarray())    
    pcd_name.paint_uniform_color(cd.NAVY.as_ndarray())    
    '''
