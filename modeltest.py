#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Acquisition and Processing of 3D Geometry - Coursework 3
# Project 13 - Learning to Estimate Normals
# Student Name: William Herbosch / Sim Zi Jian
# Lecturer: Niloy Mitra
# 2018-19

#Imports
import open3d
from open3d import read_point_cloud as readPointCloud, write_point_cloud as writePointCloud, draw_geometries as draw, PointCloud, Vector3dVector, evaluate_registration
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.decomposition import PCA as principle_component_analysis
from sklearn.model_selection import train_test_split as tts
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
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
    #(have the network train on chunks of the data rather than a whole)
    global batch_size
    batch_size = 128
    #the number of iterations the network trains for
    global epochs
    epochs = 25
    ###############################################
    #Read in data
    print("Preprocessing...")
    #Mesh to work on
    global Mesh
    #Load Mesh
    #Bunny:
    Mesh = readPointCloud("bunny.ply")

#Method for displaying an accumulator
def display_accumulator(accumulator_set, this_accumulator, display_red):
    #print(accumulator_set[this_accumulator])
    max_inten = np.amax(accumulator_set[this_accumulator])
    x = accumulator_set[this_accumulator] * (255/max_inten)
    x = np.abs(x - 255)
    img = Image.fromarray(x)
    if (display_red == True):
       img = img.convert('RGBA')
       data = np.array(img)
       red, green, blue, alpha = data.T
       black_areas = (red == 0) & (blue == 0) & (green == 0)
       data[..., :-1][black_areas.T] = (255, 0, 0)
       img = Image.fromarray(data)
    return img

#Normalize normal method
def normalize(this_n):
    if(np.linalg.norm(this_n) == 0):
        return this_n
    normalized_n = this_n/np.linalg.norm(this_n)
    return normalized_n

#Convert Open3d Point Cloud to Numpy Array
def convert_PC2NA(mesh_as_PC):
    output = (np.asarray(mesh_as_PC.points))
    return output

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
    vote_normals = np.zeros((numberOfPoints, 3))
    #search the point cloud
    for this_point in range(len(indices)):
        print(this_point+1)
        #store
        triplets = []
        vote_normal_accumulator = np.zeros((size_M, size_M, 3))
        max_vote_number = 0
        x_comp_ofmax = 0
        y_comp_ofmax = 0
        #for numberOfTriplets to be made
        for this_triplet in range(int(numberOfTriplets)):
            #obtain 3 random neighbours
            triplet = random.sample(list(indices[this_point]), 3)
            #add to triplets
            triplets.append(triplet)
        #calculate PCA
        neighbourhoodPoints = []
        for neighbours in indices[int(this_point)]:
            neighbourhoodPoints.append(list(Mesh_pointCloud[neighbours]))
        #pca3d_covariance_matrix = pca_3d_covariance_matrix(neighbourhoodPoints)
        #pca2d_covariance_matrix = pca_2d_covariance_matrix(neighbourhoodPoints)
        #transformed_mesh_pointArray = Mesh_pointArray.dot(pca3d_covariance_matrix)
        ###################
        transformed_mesh_pointArray = Mesh_pointArray
        ###################
        #For each triplet, calculate the normal of the plane that they span
        normals = []
        for this_triplet in range(len(triplets)):
            #obtain points
            p1 = transformed_mesh_pointArray[triplets[this_triplet][0]]
            p2 = transformed_mesh_pointArray[triplets[this_triplet][1]]
            p3 = transformed_mesh_pointArray[triplets[this_triplet][2]]
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            n = normalize(n)
            referencePoint = [0,0,1]
            if (np.dot(referencePoint, n) < 0):
                n *= -1
            #transformed_normal = n.dot(pca2d_covariance_matrix)
            #Compute x and y components for the accumulator
            x_comp = math.floor(((n[0] + 1)/2) * size_M)
            y_comp = math.floor(((n[1] + 1)/2) * size_M)
            #For the h case where x or y comp = 1
            if (n[0] >= 1):
                x_comp = size_M - 1
            if (n[1] >= 1):
                y_comp = size_M - 1
            #add vote
            accumulator[int(this_point)][int(x_comp)][int(y_comp)] += 1
            #add normals
            vote_normal_accumulator[int(x_comp)][int(y_comp)] += n
            if(accumulator[int(this_point)][int(x_comp)][int(y_comp)] > max_vote_number):
                max_vote_number = accumulator[int(this_point)][int(x_comp)][int(y_comp)]
                x_comp_ofmax = x_comp
                y_comp_ofmax = y_comp
        mean_normal = normalize(vote_normal_accumulator[int(x_comp_ofmax)][int(y_comp_ofmax)])
        vote_normals[int(this_point)] = mean_normal
    return accumulator, vote_normals

#Main
def main():

    preprocess()
    #Set up
    random.seed(0)
    print("Filling accumulators...")
    Mesh = readPointCloud("bunny.ply")
    Mesh_copy = copy.deepcopy(Mesh)
    #accumulator_filled = np.load("accumulator_filled_after_reorientation.dat", allow_pickle=True)

    accumulator_filled, vote_normals = hough_transform(Mesh_copy)
    #OPTIONAL display an image
    the_chosen = 0
    display_red = True
    show_this = display_accumulator(accumulator_filled, the_chosen, display_red)
    imshow(show_this)
    #STEP 3 Train the network
    training_vertex_normals = Mesh_copy.normals
    referencePoint = [0,0,1]
    for i in range(len(vote_normals)):
        if(np.dot(referencePoint, vote_normals[i]) < 0):
            vote_normals[i] *= -1
    for i in range(len(training_vertex_normals)):
        training_vertex_normals[i] = normalize(training_vertex_normals[i])
        if (np.dot(referencePoint, training_vertex_normals[i]) < 0):
            training_vertex_normals[i] *= -1
    #Begin training
    training_vertex_normals = np.delete(training_vertex_normals, 2, 1)
    model = load_model("accu_model_after_reorientation.h5")
    #validation method
    test_x = accumulator_filled[:2000]
    test_y = training_vertex_normals[:2000]
    test_x = test_x.reshape(test_x.shape[0], 33, 33, 1)
    predictions = model.predict(test_x, batch_size = batch_size)
    for i in range(400):
        rand = random.randint(0, 1999)
        print("Number:", i)
        print("True: ", test_y[rand])
        print("Prediction: ", predictions[rand])
        print("Mean normal in most voted bin: ", vote_normals[rand])
        print("Squared error: ", np.linalg.norm(np.square(test_y[rand]-predictions[rand])))

#Begins the program by running Main method
if __name__ == '__main__':
    main()
