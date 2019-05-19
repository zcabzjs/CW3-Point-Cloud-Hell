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

#Main
def main():

    batch_size = 128
    #Set up
    random.seed(0)
    print("Filling accumulators...")
    Mesh = readPointCloud("dragon.ply")
    Mesh_copy = copy.deepcopy(Mesh)
    accumulator_filled = np.load("accumulator_filled.dat", allow_pickle=True)


    #OPTIONAL display an image
    the_chosen = 0
    display_red = True
    show_this = display_accumulator(accumulator_filled, the_chosen, display_red)
    imshow(show_this)
    #STEP 3 Train the network
    training_vertex_normals = Mesh_copy.normals
    for i in range(len(training_vertex_normals)):
        training_vertex_normals[i] = normalize(training_vertex_normals[i])
    #Begin training
    model = load_model("accu_model.h5")
    #validation method
    test_x = accumulator_filled[:200]
    test_y = training_vertex_normals[:200]
    test_x = test_x.reshape(test_x.shape[0], 33, 33, 1)
    predictions = model.predict(test_x, batch_size = batch_size)
    for i in range(5):
        rand = random.randint(0, 199)
        print("Number:", i)
        print("True: ", test_y[rand])
        print("Prediction: ", predictions[rand])

#Begins the program by running Main method
if __name__ == '__main__':
    main()
