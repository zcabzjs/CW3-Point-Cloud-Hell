import numpy as np
import math
import copy
from open3d import read_point_cloud as readPointCloud, write_point_cloud as writePointCloud, draw_geometries as draw, PointCloud, Vector3dVector, evaluate_registration, draw_geometries

#Convert Open3d Point Cloud to Numpy Array
def convert_PC2NA(mesh_as_PC):
    output = (np.asarray(mesh_as_PC.points))
    return output

#Convert Numpy Array to Open3d Point Cloud
def convert_NA2PC(mesh_as_NA):
    output = PointCloud()
    output.points = Vector3dVector(mesh_as_NA)
    return output

def loadFile():
    global M1
    M1 = readPointCloud("example_meshes/cube.ply")

def normalise(v):
    if(np.linalg.norm(v) == 0):
        return v;
    normalised_v = v/np.linalg.norm(v)
    return normalised_v

def normalise_array(array):
    for index in xrange(len(array)):
        array[index] = normalise(array[index])
    return array

#Define the main method
def main():
    #Set up
    loadFile()

    #make a copy of the mesh
    M1_copy = copy.deepcopy(M1)
    #creates an array of points from the mesh
    M1_pointArray = convert_PC2NA(M1_copy)
    M1_pointCloud = Vector3dVector(M1_pointArray)

    V = M1_pointCloud[0]
    print(M1_pointArray)

    print(np.linalg.norm(M1_pointArray[0]))
    print(normalise_array(M1_pointArray))


#Begins the program by running Main method
if __name__ == '__main__':
    main()
