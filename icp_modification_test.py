# Author:  Reilly Bova '20
# Date:    13 November 2018
# Project: COS 526 A2 — Point Cloud Registration
#
# File:    icp.py
# About:   Implements the Iterative Closest Points algorithm
#          Takes 2 *.pts files as the argument and tries to align the points of
#          the first file with those in the second. Expects *.xf to exist for
#          each file as well. Output appears in ./results

import open3d as o3d
import os
import numpy as np
from sys import argv
from random import shuffle
from math import *
from lib.utils import *
from lib.kdtree import KdTree

# Check usage
#if (len(argv) != 3):
#    print("Usage Error: icp.py takes additional two arguments.\n"
#            "Proper usage is \"icp.py file1.pts file2.pts\"")
#    quit()

# Check if pts files exist
file1 = argv[1] #source points(dataset)
file2 = argv[2] #target points(dataset)


#if (not os.path.isfile(file1)):
#    print("Error: Could not find .pts file: " + file1)
#    quit()
#if (not os.path.isfile(file2)):
#    print("Error: Could not find .pts file: " + file2)
#    quit()

#check file extension, type
#file1_xf = '.'.join(file1.split('.')[:-1]) + '.xf'
#file2_xf = '.'.join(file2.split('.')[:-1]) + '.xf'

#file1_xf = '.'.join(file1.split('.')[:-1]) + '.xf'
#file1_xf = '.'.join(file1.split('.')[2:3])
file1_xf = file1.split('.')[-1]
#file1_no_extension = '.'.join(file1.split('.')[:-1])
file1_no_extension = file1.split('/')[-1].split('.')[-2]
print(file1_xf)
print(file1_no_extension)


# Load pts
#pts1 = load_pts(file1)
#pts2 = load_pts(file2)
#pts3 = load_ply('/home/atom/Workplace/reactor_head_scan/scan01/scan01.ply')
#pts3 = []
#pts3.append([1,2,3])
#pts3.append([1,2,3])
#print(pts3)
#print(pts1[0].s)
#print(pts1[0].n)
#print(len(pts1))
#print(len(pts2))
#print(len(pts3))
#print(pts3[0].s)
#print(pts3[0].n)

#print(divmod(1,5))

# All finished


# Check usage
#if (len(argv) != 6):
#    print("number of arguments: ", len(argv))
#    print("Usage Error: icp.py takes additional four arguments.\n"
#            "Proper usage is \"icp.py file1.pts file2.pts NP SC NS\"")
#    quit()

# Check if pts files exist
file1 = argv[1] #source points(dataset)
file2 = argv[2] #target points(dataset)
NP = argv[3] #number of samples in the computation
SC = argv[4] #scaling number, once every SC points will be sampled for use in case there are too many points in the point cloud 
NS = argv[5] #number of sample, instead of SC
#file1 = '/home/atom/Workplace/reactor_head_scan/scan01/scan01.ply'
#file2 = '/home/atom/Workplace/reactor_head_scan/scan02/scan02.ply'

#check file type
file1_type = file1.split('.')[-1]
file2_type = file2.split('.')[-1]

if file1_type == 'pts' and file2_type == 'pts':
    if (not os.path.isfile(file1)):
        print("Error: Could not find .pts file: " + file1)
        quit()
    if (not os.path.isfile(file2)):
        print("Error: Could not find .pts file: " + file2)
        quit()
    # Load pts
    pts1 = load_pts(file1)
    pts2 = load_pts(file2)
elif file1_type == 'ply' and file2_type == 'ply':
    if (not os.path.isfile(file1)):
        print("Error: Could not find .ply file: " + file1)
        quit()
    if (not os.path.isfile(file2)):
        print("Error: Could not find .ply file: " + file2)
        quit()
    # Load pts
    pts1 = load_ply(file1,int(SC),int(NS))
    pts2 = load_ply(file2,int(SC),int(NS))
elif file1_type != file2_type:
    print("Error: Input file types are different! ")
    quit()

elif file1_type != 'pts' or file1_type != 'ply' or file2_type != 'pts' or file2_type != 'ply':
    print("Unknown input file type: use either pts or ply only")
    quit()
quit()
