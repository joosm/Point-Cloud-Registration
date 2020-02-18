import numpy as np
#from open3d import *
import open3d as o3d

import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from os.path import expanduser


def custom_draw_geometry(pcd):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


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

    #pcd = o3d.io.read_point_cloud("../data/robot1.pcd")
    home = expanduser("~")    
    pcd_data = 'scan03'
    pcd = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan/'+pcd_data+'/'+pcd_data+'.ply',format='ply')  
    print("Compute the normals of the downsampled point cloud")
    voxel_size = 1 #original = 0.01
    downpcd = pcd.voxel_down_sample(voxel_size = voxel_size)
    downpcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(
            radius = 0.02, max_nn = 30))
    print("Downsampled point cloud shape: ")
    print(np.asarray(downpcd.points).shape)
    print(np.asarray(downpcd.colors).shape)
    #o3d.visualization.draw_geometries([downpcd])

    print("Near-origin points removal")
    distance_criteria = 30 #30mm from the origin (0,0,0)
    near_origin_removed_pcd, ind_nor = remove_near_origin_points(downpcd,distance_criteria)
    display_inlier_outlier(downpcd, ind_nor)

    print("Statistical oulier removal")
    cl, ind = near_origin_removed_pcd.remove_statistical_outlier(nb_neighbors=30,
                                                         std_ratio=1.0)

    display_inlier_outlier(near_origin_removed_pcd, ind)
    #display_inlier_outlier(voxel_down_pcd, ind)

    print("Radius oulier removal")
    clean_pcd, ind = near_origin_removed_pcd.remove_radius_outlier(nb_points=16, radius=voxel_size*6.0)
    display_inlier_outlier(near_origin_removed_pcd, ind)    

    
    radius_normal = voxel_size * 2
    #radius_normal = voxel_size * 3
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    clean_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))


    radius_feature = voxel_size * 5 #original = 5
    print("Compute FPFH feature with search radius %.3f." % radius_feature)
    fpfh = o3d.registration.compute_fpfh_feature(clean_pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
    print("FPFH shape: ")
    print(fpfh.data.shape)
    print(fpfh.data.shape[0])
    #length  = 2747
    length = 100
    xpos, ypos = np.meshgrid( np.arange(0,11,1), np.arange(0,length,1), indexing="ij")
    xposition = xpos.ravel()
    yposition = ypos.ravel()
    zposition = 0
    dx = dy = 0.5*np.ones_like(zposition)

    ff_dz = fpfh.data[0:11,0:length].ravel()
    sf_dz = fpfh.data[11:22,0:length].ravel()
    tf_dz = fpfh.data[22:33,0:length].ravel()    

    #print(xpos)
    #print(xpos.ravel())
    #print(ypos.ravel())
    fig1 = plt.figure(num=1)
    ax1 = fig1.add_subplot(131, projection='3d')
    ax1.bar3d(xposition,yposition,zposition,dx,dy,ff_dz)

    ax2 = fig1.add_subplot(132, projection='3d')
    ax2.bar3d(xposition,yposition,zposition,dx,dy,sf_dz)    

    ax3 = fig1.add_subplot(133, projection='3d')
    ax3.bar3d(xposition,yposition,zposition,dx,dy,tf_dz)   
    plt.show()
    
    feature_color_type1 = []
    feature_color_type2 = []
    for i in range(fpfh.data.shape[1]):
        first_feature = fpfh.data[0:11,i]
        ff_max_value = np.max(first_feature)
        #max_index = np.where(my_list == max_value)
        ff_max_index = np.argmax(first_feature)

        second_feature = fpfh.data[11:22,i]
        sf_max_value = np.max(second_feature)
        #max_index = np.where(my_list == max_value)
        sf_max_index = np.argmax(second_feature)

        third_feature = fpfh.data[22:33,i]
        tf_max_value = np.max(third_feature)
        #max_index = np.where(my_list == max_value)
        tf_max_index = np.argmax(third_feature)
        #max_value, max_index = max(enumerate(my_list), key=operator.itemgetter(1))
        #print(np.sum(fpfh.data[0:11,i]), ' max_value: ',max_value,' index: ',max_index)
        #print(np.sum(fpfh.data[22:33,i]))
        feature_color_type1.append([ff_max_value/200.0,sf_max_value/200,tf_max_value/200])
        feature_color_type2.append([ff_max_index/11,sf_max_index/11,tf_max_index/11])
        #downpcd.colors[0:3,i] = o3d.utility.Vector3dVector([ff_max_index/11,sf_max_index/11,tf_max_index/11])
        #downpcd.colors[0:3,i] = o3d.utility.Vector3dVector([ff_max_index/11,sf_max_index/11,tf_max_index/11])
    clean_pcd.colors = o3d.utility.Vector3dVector(feature_color_type1)        
    o3d.visualization.draw_geometries([clean_pcd])
    
    clean_pcd.colors = o3d.utility.Vector3dVector(feature_color_type2)        
    o3d.visualization.draw_geometries([clean_pcd])
    #hist, xedges, yedges = np.histogram2d(fpfh.data[:,0:100])#, cmap="gray")
    
    #plt.imshow(fpfh.data[:,0:100])#, cmap="gray")
    #print("test imshow")
    

