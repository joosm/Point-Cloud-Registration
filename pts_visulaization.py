import open3d as o3d
import numpy as np
from lib.utils import *
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_original = copy.deepcopy(source)

    source_original.paint_uniform_color([0,0,1]) #blue
    source_temp.paint_uniform_color([1, 0.706, 0]) #yellow, blue --> yellow transformed source to target
    #target_temp.paint_uniform_color([0, 0.651, 0.929])
    target_temp.paint_uniform_color([1, 0, 0]) #red, target
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_original, source_temp, target_temp])

def draw_registration_result2(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

#source = o3d.io.read_point_cloud("../Point-Cloud-Registration/output/bun045.pts",format='xyzn')
#target = o3d.io.read_point_cloud("../Point-Cloud-Registration/output/bun000.pts",format='xyzn')
#source_xf = load_xf("../Point-Cloud-Registration/output/bun045.xf")
#target_xf = load_xf("../Point-Cloud-Registration/output/bun000.xf")
'''
data1 = o3d.io.read_point_cloud('/home/atom/Workplace/reactor_head_scan/scan01/scan01.ply',format='ply')
data2 = o3d.io.read_point_cloud('/home/atom/Workplace/reactor_head_scan/scan02/scan02.ply',format='ply')
source = data1.voxel_down_sample(voxel_size=4.0)
target = data2.voxel_down_sample(voxel_size=4.0)
'''

source = o3d.io.read_point_cloud("../Point-Cloud-Registration/output/scan02.pts",format='xyzn')
target = o3d.io.read_point_cloud("../Point-Cloud-Registration/output/scan03.pts",format='xyzn')
source_xf = load_xf("../Point-Cloud-Registration/output/scan02.xf")
target_xf = load_xf("../Point-Cloud-Registration/output/scan03.xf")

#o3d.visualization.draw_geometries([source_temp, target_temp])
#draw_registration_result(source, target, source_xf)
draw_registration_result(source, target, source_xf)

#draw_registration_result(source, target, target_xf)
#draw_registration_result(target, source, target_xf)
#draw_registration_result(target, source, source_xf)
#draw_registration_result(source, target, np.linalg.inv(source_xf))

#trans_init = np.asarray([[1.0, 0.0, 0.0, 0.1],
#                        [0.0, 1.0, 0.0, 0.2],
#                        [0.0, 0.0, 1.0, 0.2], [0.0, 0.0, 0.0, 1.0]])   