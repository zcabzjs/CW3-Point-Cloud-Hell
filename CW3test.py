#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Acquisition and Processing of 3D Geometry - Coursework 3
# Project 13 - Learning to Estimate Normals
# Student Name: William Herbosch / Sim Zi Jian
# Lecturer: Niloy Mitra
# 2018-19

#Imports
from open3d import read_point_cloud as readPointCloud, write_point_cloud as writePointCloud, draw_geometries as draw, PointCloud, Vector3dVector, evaluate_registration
from sklearn.neighbors import NearestNeighbors as NN
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import copy
import math 
import random
import sys


#Preprocess 
def preprocess():
    ###############################################   
    #The size of the filled accumulator (i.e. MxM)
    global size_M
    size_M = 33
    #Confidence level, a value ]0,1[  
    global confidence_interval 
    confidence_interval = 0.95
    #Max distance between true and measured distribution (the lower the better), a value ]0,1[
    global epsilon
    epsilon = 0.073
    #number of neighbours 
    global neighbourhood_size
    neighbourhood_size = 100
    ###############################################   
    #Read in data
    print("Preprocessing...")    
    #Mesh to work on
    global Mesh
    #Load Mesh
    #Bunny:
    Mesh = readPointCloud("bunny.ply")
    #Camel
    #Mesh = trimesh.load("example_meshes/camel.obj")
    #Cow
    #Mesh = trimesh.load("example_meshes/cow.obj")


#Convert Open3d Point Cloud to Numpy Array
def convert_PC2NA(mesh_as_PC):
    output = (np.asarray(mesh_as_PC.points))
    return output

#Convert Numpy Array to Open3d Point Cloud
def convert_NA2PC(mesh_as_NA):
    output = PointCloud()
    output.points = Vector3dVector(mesh_as_NA)
    return output

#Normalize normal method
def normalize(this_n):
    if(np.linalg.norm(this_n) == 0):
        return this_n
    normalized_n = this_n/np.linalg.norm(this_n)
    return normalized_n

#Methods/Functions
def hough_transform(this_mesh):
    #Convert into narray
    Mesh_pointArray = convert_PC2NA(this_mesh)
    numberOfPoints = len(Mesh_pointArray)
    #Produce PointCloud
    Mesh_pointCloud = Vector3dVector(Mesh_pointArray)
    #compute NN for this mesh 
    #We search for more than 3 neighbours because it generates more triplets
    nn_Mesh = NN(n_neighbors=neighbourhood_size, algorithm='kd_tree').fit(Mesh_pointCloud)
    distances, indices = nn_Mesh.kneighbors(Mesh_pointCloud)
    #Compute number of triplets to be made (equation 4 from paper)
    numberOfTriplets = math.ceil((1/(2*(epsilon**2)))*math.log((2*size_M*size_M)/(1-confidence_interval)))
    #Make 3D matrix
    accumulator = np.zeros((numberOfPoints, size_M, size_M))
    #search the point cloud
    for this_point in range(len(indices)):
        #store   
        triplets = []
        #for numberOfTriplets to be made
        for this_triplet in range(numberOfTriplets):
            #obtain 3 random neighbours
            triplet = random.sample(list(indices[this_point]), 3)
            #add to triplets
            triplets.append(triplet)
        #For each triplet, calculate the normal of the plane that they span 
        normals = []
        for this_triplet in range(len(triplets)):
            #obtain points
            p1 = Mesh_pointCloud[triplets[this_triplet][0]]
            p2 = Mesh_pointCloud[triplets[this_triplet][1]]
            p3 = Mesh_pointCloud[triplets[this_triplet][2]]
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            n = normalize(n)
            #if (np.dot(p1, n) > 0):
                #n = -n
            normals.append(list(n))
        #for this point in the accumulator 
        for this_normal in range(len(normals)):
            #Compute x and y components for the accumulator 
            x_comp = math.floor(((normals[this_normal][0] + 1)/2) * size_M)
            y_comp = math.floor(((normals[this_normal][1] + 1)/2) * size_M)
            #add vote
            accumulator[this_point][x_comp][y_comp] = accumulator[this_point][x_comp][y_comp] + 1
        #print(accumulator[this_point])
        #max_inten = np.amax(accumulator[this_point])
        #img = Image.fromarray(accumulator[this_point] * (255/max_inten))
        #imshow(img)
        #sjbvhsdlkh 
    return accumulator

#Main
def main():
    #Set up
    random.seed(0)
    np.set_printoptions(threshold = sys.maxsize)
    
    preprocess()
    # Create copies of meshs
    Mesh_copy = copy.deepcopy(Mesh)
    # Colour in meshes
    # RGB values were computed and determined based on the corresponding colours found in Meshlab for BunnyTask1_ManualFit)
    Mesh_copy.paint_uniform_color([192/255, 192/255, 192/255]) #Grey
    #draw
    print("Initial positions")
    #draw([Mesh_copy])
    #STEP 1, using a Hough transformation, convert PointCloud to filled accumulator (i.e. a 2D array)
    accumulator_filled = hough_transform(Mesh_copy)
    #
    print(accumulator_filled[0])
    max_inten = np.amax(accumulator_filled[0])
    img = Image.fromarray(accumulator_filled[0] * (255/max_inten))
    imshow(img)

#Begins the program by running Main method
if __name__ == '__main__':
    main()

