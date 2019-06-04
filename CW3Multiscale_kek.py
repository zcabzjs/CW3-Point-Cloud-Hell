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
    epochs = 10
    ###############################################
    #Read in data
    print("Preprocessing...")
    #Mesh to work on
    global Mesh
    #Load Mesh
    #Bunny:
    Mesh = readPointCloud("bunny.ply")
    #Implement PCA
    global run_PCA
    run_PCA = True
    scale_number = 1
    global neighbourhood_sizes
    neighbourhood_sizes = [neighbourhood_size]
    if scale_number == 3:
        neighbourhood_sizes = [neighbourhood_size, math.floor(neighbourhood_size/2), math.floor(neighbourhood_size*2)]

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

def pca_3d_adjustment_matrix(this_neighbourhood):
    cov = np.zeros([3,3])
    mean = np.mean(this_neighbourhood, axis = 0)
    for neighbour in this_neighbourhood:
        v = neighbour - mean
        cov += np.outer(v,v)
    u, s, vt = np.linalg.svd(cov)
    return vt

def pca_2d_adjustment_matrix(this_normals):
    cov = np.zeros([3,3])
    readjusted_normals = []
    for normal in this_normals:
        nx = ((normal[0] + 1)/2) * size_M
        ny = ((normal[1] + 1)/2) * size_M
        readjusted_normal = [nx,ny,0]
        readjusted_normals.append(list(readjusted_normal))
    mean = np.mean(readjusted_normals, axis = 0)
    for readjusted_normal in readjusted_normals:
        v = readjusted_normal - mean
        cov += np.outer(v,v)
    u, s, vt = np.linalg.svd(cov)
    return vt

#Methods/Functions
def hough_transform(this_mesh):
    #Convert into narray
    Mesh_pointArray = convert_PC2NA(this_mesh)
    numberOfPoints = len(Mesh_pointArray)
    PCA_matrices = np.zeros([numberOfPoints,3,3])
    #Produce PointCloud
    Mesh_pointCloud = Vector3dVector(Mesh_pointArray)
    neighbourhood_distances = []
    neighbourhood_indices = []
    print(len(neighbourhood_sizes))
    for i in range(len(neighbourhood_sizes)):
        #compute NN for this mesh
        #We search for more than 3 neighbours because it generates more triplets
        nn_Mesh = NN(n_neighbors=neighbourhood_sizes[i], algorithm='kd_tree').fit(Mesh_pointCloud)
        distances, indices = nn_Mesh.kneighbors(Mesh_pointCloud)
        neighbourhood_distances.append(distances)
        neighbourhood_indices.append(indices)
    #Compute number of triplets to be made (equation 4 from paper)
    numberOfTriplets = math.ceil((1/(2*(epsilon**2)))*math.log((2*size_M*size_M)/(1-confidence_interval)))
    #Make 3D matrix
    accumulator = np.zeros((numberOfPoints, size_M, size_M, len(neighbourhood_sizes)))
    vote_normals = np.zeros((numberOfPoints,len(neighbourhood_sizes), 3))
    #search the point cloud
    for this_point in range(numberOfPoints):
        print(this_point+1)
        #Initializing the matrices for PCA (different for each point)
        pca_2d = np.zeros([3,3])
        pca_3d = np.zeros([3,3])
        for neighbourhood_size_index in range(len(neighbourhood_sizes)):
            #store
            triplets = []
            vote_normal_accumulator = np.zeros((size_M, size_M, 3))
            max_vote_number = 0
            x_comp_ofmax = 0
            y_comp_ofmax = 0
            #for numberOfTriplets to be made
            for this_triplet in range(int(numberOfTriplets)):
                #obtain 3 random neighbours
                triplet = random.sample(list(neighbourhood_indices[neighbourhood_size_index][this_point]), 3)
                #add to triplets
                triplets.append(triplet)
            #Only calculating for 1 neighbourhood size since the others will be almost the same, and the computation is not worth it
            if(neighbourhood_size_index == 0):
                #calculate PCA
                neighbourhoodPoints = []
                for neighbours in neighbourhood_indices[neighbourhood_size_index][int(this_point)]:
                    neighbourhoodPoints.append(list(Mesh_pointCloud[neighbours]))
                if(run_PCA):
                    pca_3d = pca_3d_adjustment_matrix(neighbourhoodPoints)
            #For each triplet, calculate the normal of the plane that they span
            normals = []
            for this_triplet in range(len(triplets)):
                #obtain points
                p1 = Mesh_pointArray[triplets[this_triplet][0]]
                p2 = Mesh_pointArray[triplets[this_triplet][1]]
                p3 = Mesh_pointArray[triplets[this_triplet][2]]
                v1 = p2 - p1
                v2 = p3 - p1
                n = np.cross(v1, v2)
                if(run_PCA):
                    n = np.dot(pca_3d, n)
                #Normalise the normal
                n = normalize(n)
                #Reorientate the normal
                referencePoint = [0,0,1]
                if (np.dot(referencePoint, n) < 0):
                    n *= -1
                normals.append(list(n))
            if(neighbourhood_size_index == 0):
                if(run_PCA):
                    pca_2d = pca_2d_adjustment_matrix(normals)
                    PCA_for_this_point = np.dot(pca_2d, pca_3d)
                    PCA_matrices[this_point] = PCA_for_this_point
            for this_normal in range(len(normals)):
                if(run_PCA):
                    normals[this_normal] = np.dot(pca_2d, normals[this_normal])
                #Normalise the normal
                normals[this_normal] = normalize(normals[this_normal])
                #Reorientate the normal
                referencePoint = [0,0,1]
                if (np.dot(referencePoint, normals[this_normal]) < 0):
                    normals[this_normal] *= -1
                #Compute x and y components for the accumulator
                x_comp = math.floor(((normals[this_normal][0] + 1)/2) * size_M)
                y_comp = math.floor(((normals[this_normal][1] + 1)/2) * size_M)
                #For the h case where x or y comp = 1
                if (normals[this_normal][0] >= 1):
                    x_comp = size_M - 1
                if (normals[this_normal][1] >= 1):
                    y_comp = size_M - 1
                #add vote
                accumulator[int(this_point)][int(x_comp)][int(y_comp)][neighbourhood_size_index] += 1
                #add normals
                vote_normal_accumulator[int(x_comp)][int(y_comp)] += normals[this_normal]
                if(accumulator[int(this_point)][int(x_comp)][int(y_comp)][neighbourhood_size_index] > max_vote_number):
                    max_vote_number = accumulator[int(this_point)][int(x_comp)][int(y_comp)][neighbourhood_size_index]
                    x_comp_ofmax = x_comp
                    y_comp_ofmax = y_comp
            mean_normal = vote_normal_accumulator[int(x_comp_ofmax)][int(y_comp_ofmax)]/accumulator[int(this_point)][int(x_comp_ofmax)][int(y_comp_ofmax)][neighbourhood_size_index]
            vote_normals[int(this_point)][neighbourhood_size_index] = mean_normal
    return accumulator, PCA_matrices

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

#Method for training the model
def train_network(filled_accumulators, training_normals):
    #input image dimensions
    img_x, img_y = size_M, size_M
    #make training set
    #Only have n_x and n_y
    training_normals = np.delete(training_normals, 2, 1)
    #Split the data
    training_x, test_x, training_y, test_y = tts(filled_accumulators, training_normals, test_size = 0.25)
    #print(training_x[0])
    #print(training_y[0])
    #print(test_x[0])
    #print(test_y[0])
    #Reshape data
    training_x = training_x.reshape(training_x.shape[0], img_x, img_y, len(neighbourhood_sizes))
    test_x = test_x.reshape(test_x.shape[0], img_x, img_y, len(neighbourhood_sizes))
    input_shape = (img_x, img_y, len(neighbourhood_sizes))
    #initialize model
    model = Sequential()
    #1st layer : Conv
    model.add(Conv2D(50, kernel_size = (3,3), activation = 'relu', input_shape = input_shape))
    #2nd layer : Conv
    model.add(Conv2D(50, kernel_size = (3,3), activation = 'relu'))
    #3rd layer : MaxPool to downscale the input in both width and height
    model.add(MaxPooling2D(pool_size = (2, 2)))
    #4th layer : Conv
    model.add(Conv2D(50, kernel_size = (3,3), activation = 'relu'))
    #5th layer : MaxPool again
    model.add(MaxPooling2D(pool_size = (2, 2)))
    #6th layer : Conv
    model.add(Conv2D(96, kernel_size = (3,3), activation = 'relu'))
    #7th layer : Flatten to represent all data thus far on a single row of data
    model.add(Flatten())
    #DROPOUT
    model.add(Dropout(0.5))
    #8th layer : Dense
    model.add(Dense(2048, activation = 'relu', input_shape = (3456,)))
    #DROPOUT
    model.add(Dropout(0.5))
    #9th layer : Dense
    model.add(Dense(1024, activation = 'relu'))
    #DROPOUT
    model.add(Dropout(0.5))
    #10th layer: Dense
    model.add(Dense(512, activation = 'relu'))
    #The final output only has 2 coordinates
    model.add(Dense(2))
    model.summary()
    model.compile(loss= 'mse', optimizer = 'adam', metrics = ['mse'])
    #fit the model
    history = model.fit(training_x, training_y,
          batch_size = batch_size,
          epochs = epochs,
          verbose = 1,
          validation_data = (test_x, test_y))
    #Evaluate
    score = model.evaluate(test_x, test_y, verbose = 0)
    #return loss
    print('Test loss:', score[0])
    #save model
    model.save("dragon_PCA_multi.h5")
    return model, test_x, test_y

#Predict for some test obejcts
def predict_this(model, test_x, test_y, validation_normals, PCA_matrices):
    predictions = model.predict(test_x, batch_size = batch_size)
    output_predictions = []
    for i in range(len(test_x)):
        print("Number:", i)
        print("True: ", test_y[i])
        z_square = 1 - test_y[i][0] * test_y[i][0] - test_y[i][1] * test_y[i][1]
        v1 = np.zeros(3)
        if(z_square < 0):
            v1 = [test_y[i][0], test_y[i][1], 0]
            v1 = normalize(v1)
        else:
            v1 = [test_y[i][0],  test_y[i][1], math.sqrt(z_square)]
        v1 = np.dot(np.linalg.inv(PCA_matrices[i]), v1)
        print("Validation normal:", validation_normals[i])
        print("Calculated normal:", v1)
        print("Prediction: ", predictions[i])
        z_square = 1 - predictions[i][0] * predictions[i][0] - predictions[i][1] * predictions[i][1]
        v2 = np.zeros(3)
        if(z_square < 0):
            v2 = [predictions[i][0], predictions[i][1], 0]
            v2 = normalize(v2)
        else:
            v2 = [predictions[i][0],  predictions[i][1], math.sqrt(z_square)]
        v2 = np.dot(np.linalg.inv(PCA_matrices[i]), v2)
        print("Predicted normal:", v2)
        print("Squared error: ", np.linalg.norm(np.square(test_y[i]-predictions[i])))
        print("Squared vector error:", np.linalg.norm(np.square(v1-v2)))
        output_predictions.append(predictions[i])
    return output_predictions

#Main
def main():
    #Set up
    random.seed(0)
    np.set_printoptions(threshold = sys.maxsize)
    #Proprocess
    preprocess()
    # Create copies of meshs
    Mesh_copy = copy.deepcopy(Mesh)
    # Colour in meshes
    # RGB values were computed and determined based on the corresponding colours found in Meshlab for BunnyTask1_ManualFit)
    Mesh_copy.paint_uniform_color([192/255, 192/255, 192/255]) #Grey
    #draw
    print("Initial positions")
    #draw([Mesh_copy])
    #STEP 1, PCA in 3D space
    #Mesh_copy = pca_3d(Mesh_copy)
    #STEP 2, using a Hough transformation, convert PointCloud to filled accumulator (i.e. a 2D array)
    print("Filling accumulators...")
    #accumulator_filled, PCA_matrices = hough_transform(Mesh_copy)
    #np.savez_compressed('bunny_PCA_multi', accum=accumulator_filled, pca=PCA_matrices)
    loaded_accumulators = np.load('bunny_PCA_multi.npz')
    accumulator_filled = loaded_accumulators['accum']
    PCA_matrices = loaded_accumulators['pca']
    #OPTIONAL display an image
    #the_chosen = 1
    #display_red = True
    #accumulator_filled_temp = accumulator_filled[:, :, :, 1]
    #show_this = display_accumulator(accumulator_filled_temp, the_chosen, display_red)
    #imshow(show_this)
    #STEP 3 Train the network
    #Get normals
    training_vertex_normals = Mesh_copy.normals
    validation_normals = Mesh.normals
    if(run_PCA):
        for i in range(len(training_vertex_normals)):
            training_vertex_normals[i] = normalize(training_vertex_normals[i])
            validation_normals[i] = normalize(validation_normals[i])
            training_vertex_normals[i]= np.dot(PCA_matrices[i], training_vertex_normals[i])
    #To check whether we should reorientate
    referencePoint = [0,0,1]
    #For each normal in the model
    for i in range(len(training_vertex_normals)):
        #normalize the norm
        training_vertex_normals[i] = normalize(training_vertex_normals[i])
        #If the z component is negative
        if(np.dot(referencePoint, training_vertex_normals[i]) < 0):
            #Negate
            training_vertex_normals[i] *= -1
    #Begin training
    print("Training network...")
    print("Number of accumulators:", len(accumulator_filled))
    print("Number of training normals:", len(training_vertex_normals))
    #model, test_x, test_y = train_network(accumulator_filled, training_vertex_normals)
    #load
    model = load_model("dragon_PCA_multi.h5")
    #model = load_model("accu_model_after_reorientation.h5")
    test_x = accumulator_filled
    test_y = np.delete(training_vertex_normals, 2, 1)
    test_x = test_x.reshape(test_x.shape[0], size_M, size_M, len(neighbourhood_sizes))
    #show original
    draw([Mesh_copy])
    output_predictions = predict_this(model, test_x, test_y, validation_normals, PCA_matrices)
    #present normals
    print("")
    output_predictions_w_z = []
    for i in range(len(output_predictions)):
        this_prediction = output_predictions[i]
        x_square = this_prediction[0] * this_prediction[0]
        y_square = this_prediction[1] * this_prediction[1]
        z_square = -x_square - y_square + 1
        #To negate from CNN
        if (z_square < 0):
            v = [this_prediction[0], this_prediction[1], 0]
            v = normalize(v)
            output_predictions_w_z.append(v)
        else:
            z = math.sqrt(z_square)
            output_predictions_w_z.append([this_prediction[0], this_prediction[1], z])
        if(run_PCA):
            print("Point:", i)
            inv = np.linalg.inv(PCA_matrices[i])
            output_predictions_w_z[i] = np.dot(inv, output_predictions_w_z[i])
            output_predictions_w_z[i] = normalize(output_predictions_w_z[i])
            #print("Predicted normal after PCA inversion", output_predictions_w_z[i])
            #print("MSE difference", np.linalg.norm(np.square(Mesh_copy.normals[i]-output_predictions_w_z[i])))
    #put predicted
    Mesh_prediction = copy.deepcopy(Mesh_copy)
    Mesh_prediction.paint_uniform_color([192/255, 192/255, 192/255])
    output_predictions_w_z = np.asarray(output_predictions_w_z)
    output_predictions_w_z = Vector3dVector(output_predictions_w_z)
    Mesh_prediction.normals = output_predictions_w_z
    draw([Mesh_prediction])
    #Angle adjustment
    angle_adjusts = []
    for this_normal in range(len(Mesh_copy.normals)):
        #print(output_predictions_w_z[3484])
        normal_prediction = output_predictions_w_z[this_normal]
        #print(Mesh_copy.normals[3484])
        normal_true = Mesh_copy.normals[this_normal]
        #compute arccos of dot product between prediction and true normals
        #A good example should be close to 1
        this_angle = np.arccos(np.dot(normal_true, normal_prediction))
        #Store in dataset
        angle_adjusts.append(this_angle)
    print(angle_adjusts)
    #write
    writePointCloud("Output_Mesh.ply", Mesh_prediction)

#Begins the program by running Main method
if __name__ == '__main__':
    main()
