#!/usr/bin/env python

# Trains on image data to learn car steering angles for a simulated track
# November 29, 2017
# Tiffany Huang

import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout

lines = []
# Load data file
with open('data3/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# Split the data into training and validation data
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Generator function loads and processes data on the fly without having to store all the data.
def generator(samples, batch_size=32):
	num_samples = len(samples)
	# Correction of steering angle measurement for using left and right camera images
	steering_correction = 0.3

	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			steering_angles = []

			for batch_sample in batch_samples:
				# Add center, left, and right images to our input data
				imageBGR_center = cv2.imread(batch_sample[0])
				# Convert to RGB because cv2 loads images in BGR, but drive.py which is the code
				# which runs the model for driving in the simulator autonomously loads images in RGB.
				imageRGB_center = cv2.cvtColor(imageBGR_center, cv2.COLOR_BGR2RGB)

				imageBGR_left = cv2.imread(batch_sample[1])
				imageRGB_left = cv2.cvtColor(imageBGR_left, cv2.COLOR_BGR2RGB)

				imageBGR_right = cv2.imread(batch_sample[2])
				imageRGB_right = cv2.cvtColor(imageBGR_right, cv2.COLOR_BGR2RGB)
				images.extend([imageRGB_center, imageRGB_left, imageRGB_right])

				steering_angle_center = float(batch_sample[3])
				steering_angle_left = steering_angle_center + steering_correction
				steering_angle_right = steering_angle_center - steering_correction
				steering_angles.extend([steering_angle_center, steering_angle_left, steering_angle_right])

				# Augment the data by flipping the images and the corresponding steering angles
				images.append(cv2.flip(imageRGB_center,1))
				images.append(cv2.flip(imageRGB_left,1))
				images.append(cv2.flip(imageRGB_right,1))
				steering_angles.append(steering_angle_center*-1.0)
				steering_angles.append(steering_angle_left*-1.0)
				steering_angles.append(steering_angle_right*-1.0)

			X_train = np.array(images)
			y_train = np.array(steering_angles)
			yield sklearn.utils.shuffle(X_train, y_train)

# Call the generator for training and validation data.
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
# Normalize the images
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# Crop the images to avoid distracting pixels
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# The Nvidia network: Bojarski, Mariusz, et al. "End to end learning for self-driving cars." 
# arXiv preprint arXiv:1604.07316 (2016).
model.add(Conv2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))
# Use mean squared error loss function and Adam optimizer
model.compile(loss='mse', optimizer='adam')

# Store the model fit as an object to plot training and validation plot
history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), 
	validation_data = validation_generator, nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)

# Save the model
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model Mean Squared Error Loss')
plt.ylabel('Mean Squared Error Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()