'''
Created on 19 Feb 2018

@author: williamherbosch

Proof-of-Concept : Convolution Neural Network for MNist dataset
This program offers an alternative architectural structure to neural network design by applying a convolutional model to the MNist dataset.
A convolutional network is a form of deep, feed-forward neural network that variations of multi-layered perceptrons to require minimal preprocessing, by having their architecture "share" weights between different layers.
What this means is that instead of taking a picture of MNist and splitting each 28x28 image into a row of pixels, a convolutional network looks at the 2D image itself and learns from the image as a whole. 
For the most part, this program is practically identical to the Dense Keras neural network, with two main distinctions
1) The way in which the different sub-datasets are initialised (i.e. trainingobjects, testobjects, traininglabels, testlabels)
2) The actual structure of the neural network to implement convolutional layers. 
This alternative neural network does take some time to run when compared to the dense network, so a workaround will try to be found. 
'''
#IMPORTS
####################################################################################
#Allows for the use of more advanced mathematical structures and calculations to be computed in Python. 
import numpy as np  					
#Import the entirty of Keras, needed for both converting the vectors of the image into a matrix
import keras
#Needed for importing the mnist dataset, a dataset of 60000 white images that are 28x28 (784) pixels of the 10 digits, along with a test set of 10000 images.
from keras.datasets import mnist
#Imports the sequential keras model. This computes a linear stack of layers similar to that in neural networks
from keras.models import Sequential 	
#Operations that are used in the layers of the network
#Dense is used for the operation "output = activation(dot(input, kernel)) + bias"
#Dropout consists of randomly setting a rate at which it converts inputs to 0 after each update. This helps prevents overfitting
#Conv2D represents a 2D convolutional layer (specifically for images (which are 2D))
#MaxPooling2D allows the network to down-sample an input representation, reducing its dimensionality and allowing for assumptions to be made about features contained within the network's individual bins.
#Flatten the inputs to form a single line. Does not effect batch size.
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
##An optimizer for compiling Keras
from keras.optimizers import RMSprop
#A 2D plotting library that displays the images in the dataset. 
import matplotlib.pyplot as mplpp
#imports the randomizer for picking different test objects
import random

#GLOBAL
####################################################################################
#Set the number of itterations the network trains for
numberOfEpochs = 20
#Set the batchSize (have the network train on chunks of the data rather than a whole). Number should preferably be divisible by training samples
batchSize = 100
#The number of outputs/classes the network can have (0-9 in this case)
numberOfClasses = 10
#Original MNist has 60K training samples, filter this down to 2K (80% training data)
ntrainingObservations = 2000
#Original MNist has 10K test samples, filter this down to 400 (20% test data)
ntestObservations = 10000

#METHODS
####################################################################################
#a method used to display a test opject in a popup window, takes in the testobject element number, the testobjects themselves and their corrisponding labels
def displayMNist(number, testObjectsFilter, testLabelsFilter):
	#Assigns the label of the corrisponding test object
	label = testLabelsFilter[number].argmax(axis=0)
	#takes the information from the test object and reshapes it into a 28x28 grid (the image)
	image = testObjectsFilter[number].reshape([28, 28])
	#Sets a title containing which test element is being presented, as well as displaying the true label
	mplpp.title("Example: %d   Label: %d"%(number, label))
	#Display the image in white on black
	mplpp.imshow(image, cmap=mplpp.get_cmap("gray"))
	mplpp.show()

#a method used for predicting the test object's label
def predict(range):
	#range is a list of 10 elements, each element represents the likelyhood of a test object being of that label (i.e. the 1st value represents how likely it is a 0)
	#Therefore, we can take these values, find which is the greates, and assume that this is the most likeliest label, thus we make it our prediction
	#If the element with the highest value in the list is the 1st element (start from range[0]), then return 0
	if max(range) == range[0]:
		return 0
	#Repeat for all numbers 0-9
	if max(range) == range[1]:
		return 1
	if max(range) == range[2]:
		return 2
	if max(range) == range[3]:
		return 3
	if max(range) == range[4]:
		return 4
	if max(range) == range[5]:
		return 5
	if max(range) == range[6]:
		return 6
	if max(range) == range[7]:
		return 7
	if max(range) == range[8]:
		return 8
	if max(range) == range[9]:
		return 9

#MAIN
####################################################################################
def main():
	#Loads the MNist dataset and assigns it to training and test observations
	#Training = 60000 rows and 784 cols - Testing = 10000 rows and 784 cols
	#784 attributes because we check each cell in the 28x28 grid of the image
	(trainingObjects, trainingLabels), (testObjects, testLabels) = mnist.load_data()
	#Reshape the data so that we have each row as a different observation (for the backend, we assume that the data format is (rows, cols, channels))
	trainingObjects = trainingObjects.reshape(trainingObjects.shape[0], 28, 28, 1)
	#Do the same for the test objects
	testObjects = testObjects.reshape(testObjects.shape[0], 28, 28, 1)
	#set the type of values in the sets as float32 (single precision float)
	trainingObjects = trainingObjects.astype("float32")
	testObjects = testObjects.astype("float32")
	#We want to work on a small section of the data, so we only look at the specified number of rows (this might create unbalance however)
	trainingObjectsFilter = trainingObjects[:ntrainingObservations, :]
	testObjectsFilter = testObjects[:ntestObservations, :]
	#Print some information to the terminal so we know what data we are focusing on
	print("trainingObjects dimensions:", trainingObjectsFilter.shape)
	print("training samples:", trainingObjectsFilter.shape[0])
	print("test samples:", testObjectsFilter.shape[0])
	#Now that we have data, we can assign their corresponding labels
	#This is done by converting a class vector of labels into a binary class matrix
	trainingLabels = keras.utils.to_categorical(trainingLabels, numberOfClasses)
	testLabels = keras.utils.to_categorical(testLabels, numberOfClasses)
	#Filter the test labels
	trainingLabelsFilter = trainingLabels[:ntrainingObservations, :]
	testLabelsFilter = testLabels[:ntestObservations, :]
	#Initialize a sequential model for the neural network
	#The Sequential model is a linear stack of layers
	#These layers act somewhat similar to hidden layers in a multi-layerd neural network
	model = Sequential()
	#Add out first (input) layer, a Conv2D
	#32 refers to the dimensionality of the output space (i.e. the number of output filters in the convolution)
	#the kernel size specifies the width and height of a 2d convolution window. 
	#Our input shape (for when we add our input later) is the dimentions of the format of the backend
	model.add(Conv2D(32, kernel_size=(3, 3), activation="sigmoid", input_shape=(28, 28, 1)))
	#An additional layer that has an output filter of double that of the previous
	model.add(Conv2D(64, (3, 3), activation="sigmoid"))
	#A maxpool layer used to downscale the input in both width and height
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#Because we do not want to risk overfitting, we drop some neurons with the least relevant information once undergone the initial layer
	model.add(Dropout(0.2))
	#A flatten layer used to represent all data thus far on a single row of data (a now 1D representation of the MNist image)
	model.add(Flatten())
	#From here, it's almost identical to the Dense neural network
	#again, for same as the Dense neural network, we'll have a value of 500
	model.add(Dense(500, activation="sigmoid"))
	#Repeat for a second time
	model.add(Dropout(0.2))
	#Formulates the outputs into a range of 0-9 values (this is the output layer)
	model.add(Dense(numberOfClasses, activation="softmax"))
	#Gives a summary of the structure of the network created (i.e. how many neurons in each layer, etc)
	model.summary()
	#Compiles the model
	#Catagorical Crossentropy refers to the loss generated between predictions and target outputs
	#Also initializes an optimiser in RMSprop, which is a learning rate method that divides the learning rate by an exponentially decaying average of squared gradients
	model.compile(loss='categorical_crossentropy', optimizer = RMSprop(), metrics = ['accuracy'])
	#Fits the model and commenses training
	history = model.fit(trainingObjectsFilter, trainingLabelsFilter, batch_size=batchSize, epochs=numberOfEpochs, verbose=1, validation_data=(testObjectsFilter, testLabelsFilter))
	#Evaluates the final learning experience of the network by providing the total loss and the accuracy of predictions
	score = model.evaluate(testObjectsFilter, testLabelsFilter, verbose=0)
	#prints loss and accuracy to terminal
	print("Test loss:", score[0])
	print("Test accuracy:", score[1])
	#Uses the now trained model on the test dataset in batches of 10 and predicts labels for each object
	predictions = model.predict(testObjectsFilter, batch_size = 10)
	#For 20 different test objects
	for i in range(2):
		rand = random.randint(0,199)
		#Print the predicted output to the terminal
		print("Prediction: ",predict(predictions[rand]))
		#Display the target output in a popup window
		displayMNist(rand, testObjectsFilter, testLabelsFilter)

#Begins the program by running Main method
if __name__ == '__main__':
    main()