from scipy.misc import imread
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, GlobalAveragePooling2D, Conv2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from resizeimage import resizeimage
import numpy as np
from glob import glob
from skimage import color
from tqdm import tqdm
import cv2
import os

classes = {
0 : 'Speed limit = 20',
1 : 'Speed limit = 30',
2 : 'Speed limit = 50',
3 : 'Speed limit = 60',
4 : 'Speed limit = 70',
5 : 'Speed limit = 80',
6 : 'Speed limit = 90',
7 : 'Speed limit = 100',
8 : 'Speed limit = 120',
9 : 'Speed limit = 70',
10 : 'Speed limit = 70',
11 : 'Speed limit = 70',
12 : 'Speed limit = 70',
13 : 'Speed limit = 70',
 }


def to_grayscale(img):
    image = cv2.imread(img)
    image = cv2.resize(image,(int(32),int(32)))
    im = np.uint8(image)
    gray = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2GRAY)
    return np.array(gray)

def load_images(files_path):
    data = load_files(files_path) #load files
    images = np.array(data['filenames']) #load images
    labels = np_utils.to_categorical(np.array(data['target']), 62) #one hot encoding the labels
    return images, labels

# def path_to_tensor(img_path):
#     # loads RGB image as PIL.Image.Image type
#     img = image.load_img(img_path, target_size=(32, 32))
#     # convert PIL.Image.Image type to 3D tensor with shape (28, 28, 3)
#     x = image.img_to_array(img)
#     # print('img shape: ', x.shape[:])
#     # convert 3D tensor to 4D tensor with shape (1, 28, 28, 1) and return 4D tensor
#     return np.expand_dims(x, axis=0)

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    # img = image.load_img(img_path, target_size=(32, 32))
    # convert PIL.Image.Image type to 3D tensor with shape (28, 28, 3)
    x = image.img_to_array(img_path)
    # print('img shape: ', x.shape[:])
    # convert 3D tensor to 4D tensor with shape (1, 28, 28, 1) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

train_images, train_labels = load_images(r"D:\Courses\Udacity Nanodegrees\Machine Learning Nanodegree\Capstone Project\Traffic signs dataset\BelgiumTSC_Training\Training")
test_images, test_labels = load_images(r"D:\Courses\Udacity Nanodegrees\Machine Learning Nanodegree\Capstone Project\Traffic signs dataset\BelgiumTSC_Testing\Testing")

gray_training_images = np.zeros(train_images.shape[:])
gray_training_images = [to_grayscale(image) for image in train_images]
gray_training_images = np.array(gray_training_images)
# gray_testing_images = np.zeros(test_images.shape[:])
# gray_testing_images = [to_grayscale(image) for image in test_images]
# print("train:", train_images[1])
# print("gray", gray_training_images[1])

# train_tensors = paths_to_tensor(train_images).astype('float32')/255
# test_tensors = paths_to_tensor(test_images).astype('float32')/255

train_tensors = paths_to_tensor(gray_training_images).astype('float32')/255
# test_tensors = paths_to_tensor(gray_testing_images).astype('float32')/255

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

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy']) #compiling our model

# checkpointer = ModelCheckpoint(filepath = r'D:\Courses\Udacity Nanodegrees\Machine Learning Nanodegree\Capstone Project\best_weights\weights.hdf5', verbose = 1, save_best_only = True)

# model.fit(train_tensors, train_labels, batch_size = 200, nb_epoch = 60, validation_split=0.2, callbacks=[checkpointer], shuffle = True)


# model.load_weights(r'D:\Courses\Udacity Nanodegrees\Machine Learning Nanodegree\Capstone Project\best_weights\weights.hdf5')

score = model.evaluate(test_tensors, test_labels, verbose = 1)
accuracy = score[1] *100 #score[0] returns loss value, score[1] returns the metrics value (accuracy)
print(r'\n\nAccuracy score: ', accuracy)

# prediction = model.predict(path_to_tensor(r"D:\Courses\Udacity Nanodegrees\Machine Learning Nanodegree\Capstone Project\Traffic signs dataset\BelgiumTSC_Testing\Testing\00000\02294_00000.ppm"))
# print(np.argmax(prediction))

# prediction = model.predict(path_to_tensor(r"D:\Courses\Udacity Nanodegrees\Machine Learning Nanodegree\Capstone Project\20.jpg"))
# print(np.argmax(prediction))
