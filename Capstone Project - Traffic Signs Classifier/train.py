from scipy.misc import imread
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, GlobalAveragePooling2D, Conv2D, Dropout, Flatten
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
import numpy as np
from glob import glob
from skimage import color
from tqdm import tqdm
import cv2
import os
from utils import load_images, path_to_tensor, paths_to_tensor

#importing the training and testing images
train_images, train_labels = load_images(r"D:\Courses\Udacity Nanodegrees\Machine Learning Nanodegree\Capstone Project\Traffic signs dataset\BelgiumTSC_Training\Training")
test_images, test_labels = load_images(r"D:\Courses\Udacity Nanodegrees\Machine Learning Nanodegree\Capstone Project\Traffic signs dataset\BelgiumTSC_Testing\Testing")

#converting the trainig and testing images to 4d tensors to be fed to the convolutional layers
train_tensors = paths_to_tensor(train_images).astype('float32')/255
test_tensors = paths_to_tensor(test_images).astype('float32')/255

#Model's architecture
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 2, padding='same', strides = 1, activation = 'relu', input_shape = train_tensors.shape[1:]))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 32, kernel_size = 2, padding='same', strides = 1, activation = 'relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 64, kernel_size = 2, padding='same', strides = 1, activation = 'relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 128, kernel_size = 2, padding='same', strides = 1, activation = 'relu'))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(62, activation = 'softmax'))
model.summary()

#compiling the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy']) #compiling our model

#making a checkpointer
# checkpointer = ModelCheckpoint(filepath = r'D:\Courses\Udacity Nanodegrees\Machine Learning Nanodegree\Capstone Project\best_weights\final.hdf5', verbose = 1, save_best_only = True)

#fitting the model
# model.fit(train_tensors, train_labels, batch_size = 100, nb_epoch = 100, validation_split=0.2, callbacks=[checkpointer], shuffle = True)

#loading the model weights
model.load_weights(r'D:\Courses\Udacity Nanodegrees\Machine Learning Nanodegree\Capstone Project\best_weights\final.hdf5')

#saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")

score = model.evaluate(test_tensors, test_labels, verbose = 1)
accuracy = score[1] *100 #score[0] returns loss value, score[1] returns the metrics value (accuracy)
print(r'\n\nAccuracy score: ', accuracy)
