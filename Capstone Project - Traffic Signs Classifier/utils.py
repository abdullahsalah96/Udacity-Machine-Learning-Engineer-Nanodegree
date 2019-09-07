from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
from skimage import color
from tqdm import tqdm
import cv2
import os
from keras.preprocessing import image


def load_images(files_path):
    """
    A funtion that takes the path of the images and returns a numpy array containing the images paths and a numpy array of one hot encoded labels
    """
    data = load_files(files_path) #load files
    images = np.array(data['filenames']) #load images
    labels = np_utils.to_categorical(np.array(data['target']), 62) #one hot encoding the labels
    return images, labels

def path_to_tensor(img_path):
    """
    A funtion that takes the path of the image and converts it into a 4d tensor to be fed to the convolutional network
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(32, 32))
    # convert PIL.Image.Image type to 3D tensor with shape (28, 28, 3)
    x = image.img_to_array(img)
    # print('img shape: ', x.shape[:])
    # convert 3D tensor to 4D tensor with shape (1, 28, 28, 1) and return 4D tensor
    return np.expand_dims(x, axis=0)



def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
