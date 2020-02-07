# Author:  Reilly Bova '20
# Date:    11 November 2018
# Project: COS 526 A2 — Point Cloud Registration
#
# File:    utils.py
# About:   Provides utility functions for reading and writing .pts and .xf files

import os
import numpy as np
from .point import Point

import open3d as o3d

from plyfile import (PlyData, PlyElement, make2d,
                     PlyHeaderParseError, PlyElementParseError,
                     PlyProperty)

from random import shuffle

from scipy.stats import special_ortho_group

# Loads data from the given .xf file into a 4x4 np matrix
def load_xf(file_name):
    with open(file_name) as f:
        data = f.read()
        rows = []
        for r in data.split('\n'):
            if (len(r) > 0):
                rows.append(r)

        if (len(rows) != 4):
            print("Error: Invalid number of rows detected in .xf file")
            rows = ["0 0 0 0" for i in range(4)]

        # Could do in one-liner, but this has error checking
        result = []
        for r in rows:
            c = []
            for v in r.split(' '):
                if (len(v) > 0):
                    c.append(float(v))
            if (len(c) != 4):
                print("Error: Invalid number of columns detected in {}".format(file_name))
                c = [0, 0, 0, 0]
            result.append(c)

    return np.matrix(result)

# # Loads data from the given .pts file into a list of Points
def load_pts(file_name):
    with open(file_name) as f:
        data = f.read()
        rows = data.split('\n')

        result = []
        for r in rows:
            if (len(r) == 0):
                continue
            pData = [float(v) for v in r.split(' ')]
            if (len(pData) != 6):
                print("Error: Insufficient data provided for a point in {}".format(file_name))
                pData = [0, 0, 0, 0, 0, 0]
            #print(pData[0:3], pData[3:6])
            #print(pData)   
            result.append(Point(pData[0:3], pData[3:6]))
    print("data of final data size = ", len(result))
    return result

def load_ply(file_name,sampling_scale, number_of_sample):
    #with PlyData.read(file_name) as f:
    f = PlyData.read(file_name)
    vertex = f['vertex']
    (x,y,z,nx,ny,nz) = (vertex[i] for i in ('x','y','z','nx','ny','nz'))
    #data = (vertex[i] for i in ('x','y','z','nx','ny','nz'))
    rows = []

    for i in range(len(x)):
        dist = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
        if dist > 150.0:
            #print(i,x[i],y[i],z[i],nx[i],ny[i],nz[i])
            rows.append([x[i],y[i],z[i],nx[i],ny[i],nz[i]])
    print("data pts size = ", len(rows))
    #rows = data.split('\n')
    temp_result = []
    #i = 1
    for r in rows:
        if (len(r) == 0):
            continue
        pData = r #[float(r) for v in r.split(' ')]
        if (len(pData) != 6):
            print("Error: Insufficient data provided for a point in {}".format(file_name))
            pData = [0, 0, 0, 0, 0, 0]
        #if i%sampling_scale == 0:
        temp_result.append(Point(pData[0:3], pData[3:6]))
        #i = i + 1
    pts_index = [i for i in range(len(temp_result))]
    #result = []
    shuffle(pts_index)
    result =  [temp_result[i] for i in pts_index[:int(number_of_sample)]]
    print('number of final data size = ',len(result))
    #print(result)
    return result    

def load_ply_shift(file_name,sampling_scale, number_of_sample):
    #with PlyData.read(file_name) as f:
    f = PlyData.read(file_name)
    vertex = f['vertex']
    (x,y,z,nx,ny,nz) = (vertex[i] for i in ('x','y','z','nx','ny','nz'))
    #data = (vertex[i] for i in ('x','y','z','nx','ny','nz'))
    rows = []

    for i in range(len(x)):
        dist = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
        if dist > 150.0:
            #print(i,x[i],y[i],z[i],nx[i],ny[i],nz[i])
            rows.append([x[i]+10,y[i]+15,z[i]-10,nx[i],ny[i],nz[i]])
    print("data pts size = ", len(rows))
    #rows = data.split('\n')
    temp_result = []
    #i = 1
    for r in rows:
        if (len(r) == 0):
            continue
        pData = r #[float(r) for v in r.split(' ')]
        if (len(pData) != 6):
            print("Error: Insufficient data provided for a point in {}".format(file_name))
            pData = [0, 0, 0, 0, 0, 0]
        #if i%sampling_scale == 0:
        temp_result.append(Point(pData[0:3], pData[3:6]))
        #i = i + 1
    pts_index = [i for i in range(len(temp_result))]
    #result = []
    shuffle(pts_index)
    result =  [temp_result[i] for i in pts_index[:int(number_of_sample)]]
    print('number of final data size = ',len(result))
    #print(result)
    return result    

# Writes the provided matrix M to the specified .xf file
def write_xf(file_name, M):
    # Make the directory if necessary
    dir = os.path.dirname(file_name)
    if (not os.path.exists(dir)):
        os.makedirs(dir)
    #if ~np.isarray(M):
    #    M = M.A

    # Write to file
    with open(file_name, "w") as f:
        for row in np.asarray(M):
        #for row in M.A:
            for val in row:
                f.write(str(val))
                f.write(' ')
            f.write('\n')
    return

# Writes the provided list of points to the specified .pts file
def write_pts(file_name, pts):
    # Make the directory if necessary
    dir = os.path.dirname(file_name)
    if (not os.path.exists(dir)):
        os.makedirs(dir)

    # Write to file
    with open(file_name, "w") as f:
        for p in pts:
            for val in (p.s + p.n):
                f.write(str(val))
                f.write(' ')
            f.write('\n')
    return


#random rotation matrix
#copy from http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
#https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices
def rand_rotation_matrix_v2(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def rand_rotation_matrix_v1(deflection=1.0):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random rotation. Small
    deflection => small perturbation.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    theta = np.random.uniform(0, 2.0*deflection*np.pi) # Rotation about the pole (Z).
    phi = np.random.uniform(0, 2.0*np.pi) # For direction of pole deflection.
    z = np.random.uniform(0, 2.0*deflection) # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    # Compute the row vector S = Transpose(V) * R, where R is a simple
    # rotation by theta about the z-axis.  No need to compute Sz since
    # it's just Vz.

    st = np.sin(theta)
    ct = np.cos(theta)
    Sx = Vx * ct - Vy * st
    Sy = Vx * st + Vy * ct
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R, which
    # is equivalent to V S - R.
    
    M = np.array((
            (
                Vx * Sx - ct,
                Vx * Sy - st,
                Vx * Vz
            ),
            (
                Vy * Sx + st,
                Vy * Sy - ct,
                Vy * Vz
            ),
            (
                Vz * Sx,
                Vz * Sy,
                1.0 - z   # This equals Vz * Vz - 1.0
            )
            )
    )
    return M


def rand_rotation_matrix_v3(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.cos(phi) * r,
        np.sin(phi) * r,
        np.sqrt(1.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = np.dot(2.0*np.outer(V, V) - np.eye(3),R)
    return M

def rand_rotation_matrix_v4(dimension=3):
    return special_ortho_group.rvs(dimension)