import numpy as np
#from open3d import *
import open3d as o3d

import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def custom_draw_geometry(pcd):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":

    pcd = o3d.io.read_point_cloud("../data/robot1.pcd")
    print("Compute the normals of the downsampled point cloud")
    voxel_size = 0.005 #original = 0.01
    downpcd = pcd.voxel_down_sample(voxel_size = voxel_size)
    downpcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(
            radius = 0.02, max_nn = 30))
    print("Downsampled point cloud shape: ")
    print(np.asarray(downpcd.points).shape)
    print(np.asarray(downpcd.colors).shape)
    #o3d.visualization.draw_geometries([downpcd])


    radius_feature = voxel_size * 5 #original = 5
    print("Compute FPFH feature with search radius %.3f." % radius_feature)
    fpfh = o3d.registration.compute_fpfh_feature(downpcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
    print("FPFH shape: ")
    print(fpfh.data.shape)
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
    feature_color = []
    for i in range(1000):
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
        feature_color.append([ff_max_value/200.0,sf_max_value/200,tf_max_value/200])
        #feature_color.append([ff_max_index/11,sf_max_index/11,tf_max_index/11])
        #downpcd.colors[0:3,i] = o3d.utility.Vector3dVector([ff_max_index/11,sf_max_index/11,tf_max_index/11])
        #downpcd.colors[0:3,i] = o3d.utility.Vector3dVector([ff_max_index/11,sf_max_index/11,tf_max_index/11])
    downpcd.colors = o3d.utility.Vector3dVector(feature_color)        
    o3d.visualization.draw_geometries([downpcd])
    #hist, xedges, yedges = np.histogram2d(fpfh.data[:,0:100])#, cmap="gray")
    
    #plt.imshow(fpfh.data[:,0:100])#, cmap="gray")
    #print("test imshow")
    

