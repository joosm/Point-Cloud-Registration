import numpy as np
import copy
import open3d as o3d
from os.path import expanduser
from os.path import isfile
#import color_dictionary as cd

from scipy import stats
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import matplotlib.cm as cm
#from matplotlib.colors import Normalize

import time


#pcd = o3d.io.read_point_cloud(point_cloud,format='ply') 

home = expanduser("~")

#[1] after registration
#pcd_data_name = 'after_registration'
#pcd_data_name = 'after_registration_two_pcds'
#pcd_data_name = 'after_registration_two_pcds_distance_10_subsampling_729points'
#pcd_data_name = 'after_registration_two_pcds_distance_5_subsampling_2576points'
#pcd_data_name = 'after_registration_two_pcds_distance_3_subsampling_6467points'
#pcd_data_name = 'after_registration_two_pcds_distance_2_subsampling_13747points'
#pcd_data_name = 'after_registration_two_pcds_distance_1_subsampling_45277points'
#pcd_data_name = 'after_registration_two_pcds_distance_05_subsampling_124556points'
#pcd_data_name = 'after_registration_two_pcds_distance_02_subsampling_173396points'

#pcd_data_name = 'after_registration_distance_3_subsampling_17679points'
#pcd_data_name = 'after_registration_distance_5_subsampling_6584points'
#pcd_data_name = 'after_registration_distance_10_subsampling_1911points'
#pcd_data_name = 'after_registration_distance_20_subsampling_518points'
#pcd_data_name = 'after_registration_random_2048_points'
#pcd_data_name = 'after_registration_random_6584_points'
#pcd_data_name = 'after_registration_random_17679_points'


#pcd_data = o3d.io.read_point_cloud(home+'/Workplace/Point-Cloud-Registration/incremental_registration_for_crashed_head/'+pcd_data_name+'.ply',format='ply')
#pcd_data = o3d.io.read_point_cloud(home+'/Workplace/Point-Cloud-Registration/incremental_registration_for_original_head/'+pcd_data_name+'.ply',format='ply')
#voxel_size = 10


#pcd_data = o3d.io.read_point_cloud(home+'/Workplace/Point-Cloud-Registration/incremental_registration_for_crashed_head/'+pcd_data_name+'.ply',format='ply')

#[2] single scan
#pcd_data_name = 'scan01_clean_downsampled'
#pcd_data_name = '0000200000'
#pcd_data_name = 'scan11'
pcd_data_name = 'scan02'
voxel_size = 2
#pcd_data = o3d.io.read_point_cloud(home+'/Workplace/Point-Cloud-Registration/pairwise_registration_for_crashed_head/'+pcd_data_name+'.ply',format='ply')
#pcd_data = o3d.io.read_point_cloud(home+'/Workplace/KAERI_Workplace1/Data/Model/stl/level1/GTCATPart/data02_300files/scan/'+pcd_data_name+'.pcd',format='pcd')
pcd_data = o3d.io.read_point_cloud(home+'/Workplace/reactor_head_scan_after_crash/'+pcd_data_name+'/'+pcd_data_name+'.ply',format='ply')

# downsampling or not
temp_pcd = pcd_data.voxel_down_sample(voxel_size)
#temp_pcd = copy.deepcopy(pcd_data)
pcd = np.asarray(temp_pcd.points)

print(pcd.shape)



start_time = time.time()
kde = stats.gaussian_kde(pcd.T)
density = kde(pcd.T)

print("density computation time: ",time.time() - start_time)
#normalize_density = normalize(density[:,np.newaxis], axis=0).ravel()
normalize_density = density/max(density)
#print(density)
#print(max(density))
#print(min(density))
#print(max(normalize_density))
#print(min(normalize_density))

#cmap = cm.autumn
cmap = cm.hot
#cmap = [[0 ,i,0] for i in normalize_density]
#print(cmap(max(normalize_density)))

color_vector = np.asarray(cmap(normalize_density))
#print(color_vector[0,:])
print(color_vector.shape)

#color_vector = np.asarray(cmap)
#print(color_vector[0,:])
#print(color_vector.shape)

#kde = stats.gaussian_kde(values)
#density = kde(values)

#fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
#x, y, z = values
#ax.scatter(x, y, z, c=density)
#plt.show()
pcd_visualize = o3d.geometry.PointCloud()
pcd_visualize.points = o3d.utility.Vector3dVector(pcd)
pcd_visualize.colors = o3d.utility.Vector3dVector(color_vector[:,0:3])

#o3d.visualization.draw_geometries([pcd_visualize])

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd_visualize)
vis.run()
vis.destroy_window()