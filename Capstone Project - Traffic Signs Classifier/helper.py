import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from tqdm import tqdm
import pandas as pd


class HelperCallbacks(tf.keras.callbacks.Callback):
    """
    A class that includers helper callback functions
    """
    def __init__(self, property = 'acc', threshold = 0.95):
        self.property = property
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get(property) > self.threshold):
            print('REACHED THRESHOLD SO STOPPING TRAINING')
            self.model.stop_training = True


class HelperImageGenerator():
    """
    A class that generates images from a given directory
    """
    def __init__(self, dir = '/home', target_size = (300, 300), batch_size = 128,  class_mode = 'binary'):
        self.generator = ImageDataGenerator(rescale = 1./255)
        self.dir = dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_mode = class_mode

    def generateImages(self):
        return self.generator.flow_from_directory(
        self.dir,
        target_size = self.target_size,
        batch_size = self.batch_size,
        class_mode = self.class_mode
        )


class HelperLoadImages():
    def load_images(self, files_path):
        """
        A funtion that takes the path of the images and returns a numpy array containing the images paths and a numpy array of one hot encoded labels
        """
        data = load_files(files_path) #load files
        images = np.array(data['filenames']) #load images
        labels = np_utils.to_categorical(np.array(data['target']), 62) #one hot encoding the labels
        return images, labels

    def path_to_tensor(self, img_path):
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

    def paths_to_tensor(self, img_paths):
        """
        A funtion that takes the path of the images and converts them into a 4d tensor to be fed to the convolutional network and normalizes them
        """
        list_of_tensors = [self.path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors).astype('float32')/255


class HelperLoadLabels():
    def loadAnnotations(self, file_path):
        """
        A function that returns the annotations file as a pandas dataframe
        """
        data = pd.read_csv(file_path, sep=";", header=None)
        data.columns = ["image", "x1", "y1", "x2", "y2", "class", "superclass", "null"]
        return data

    def getAnnotations(self, file_path):
        """
        A function that returns a numpy array that contains x1,x2,y1,y2,class
        """
        annotations = self.loadAnnotations(file_path)
        object_class = annotations.drop(['null'], axis = 1)
        labels = np.array(object_class)
        return labels

    def getLabelsForImages(self, images, annotations_file_path):
        """
        A function that return the mapped labels to the corrosponding file
        """
        annotation_labels = self.getAnnotations(annotations_file_path)
        return annotation_labels



imageLoader = HelperLoadImages()
train_images, train_labels = imageLoader.load_images(r"BelgiumTSC_Training/Training")
test_images, test_labels = imageLoader.load_images(r"BelgiumTSC_Testing/Testing")

#converting the training and testing images to 4d tensors to be fed to the convolutional layers
train_tensors = imageLoader.paths_to_tensor(train_images)
test_tensors = imageLoader.paths_to_tensor(test_images)

ann = HelperLoadLabels()
labels = ann.getLabelsForImages(train_images, "BelgiumTSD_annotations/BelgiumTSD_annotations/BTSD_training_GTclear.txt")
print(labels)
print(train_images[0])


# print(ann.getLabels("BelgiumTSD_annotations/BelgiumTSD_annotations/BTSD_training_GTclear.txt"))
# myCallback = HelperCallbacks('loss', 0.99)
# imgGen = HelperImageGenerator()
# imgLoad = HelperLoadImages()
