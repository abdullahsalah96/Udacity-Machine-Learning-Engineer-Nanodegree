
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
# train_files, train_targets = load_dataset('D:/Courses/Udacity Nanodegrees/Machine Learning Nanodegree/Projects/Project 3 - Dog Breed Classifier/dog_breed_classifier/dogImages/train')
# valid_files, valid_targets = load_dataset('D:/Courses/Udacity Nanodegrees/Machine Learning Nanodegree/Projects/Project 3 - Dog Breed Classifier/dog_breed_classifier/dogImages/valid')
# test_files, test_targets = load_dataset('D:/Courses/Udacity Nanodegrees/Machine Learning Nanodegree/Projects/Project 3 - Dog Breed Classifier/dog_breed_classifier/dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob('D:/Courses/Udacity Nanodegrees/Machine Learning Nanodegree/Projects/Project 3 - Dog Breed Classifier/dog_breed_classifier/dogImages/train/*/'))]

# print statistics about the dataset
# print('There are %d total dog categories.' % len(dog_names))
# print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
# print('There are %d training dog images.' % len(train_files))
# print('There are %d validation dog images.' % len(valid_files))
# print('There are %d test dog images.'% len(test_files))


# ### Import Human Dataset
#
# In the code cell below, we import a dataset of human images, where the file paths are stored in the numpy array `human_files`.

# In[2]:


# import random
# random.seed(8675309)

# load filenames in shuffled human dataset
# human_files = np.array(glob("lfw/*/*"))
# random.shuffle(human_files)

# print statistics about the dataset
# print('There are %d total human images.' % len(human_files))

import cv2
import matplotlib.pyplot as plt


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)




from keras.applications.resnet50 import preprocess_input, decode_predictions
#
# def ResNet50_predict_labels(img_path):
#     # returns prediction vector for image located at img_path
#     img = preprocess_input(path_to_tensor(img_path))
#     return np.argmax(ResNet50_model.predict(img))
#



### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))





from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-process the data for Keras
# train_tensors = paths_to_tensor(train_files).astype('float32')/255
# valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
# test_tensors = paths_to_tensor(test_files).astype('float32')/255


# In[13]:


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()
# print(train_tensors.shape[1:])
# print(len(dog_names))



from keras.callbacks import ModelCheckpoint


# bottleneck_fetures_Xception = np.load('/data/bottleneck_features/DogXceptionData.npz')
# valid_Xception = bottleneck_fetures_Xception['valid']
# test_Xception = bottleneck_fetures_Xception['test']
# train_Xception = bottleneck_fetures_Xception['train']
# print(train_Xception.shape)



### TODO: Define your architecture.
Xception_model = Sequential()
Xception_model.add(GlobalAveragePooling2D(input_shape = ( 7, 7, 2048)))
Xception_model.add(Dropout(0.3))
Xception_model.add(Dense(512, activation = 'relu'))
Xception_model.add(Dropout(0.3))
Xception_model.add(Dense(len(dog_names), activation='softmax'))
Xception_model.summary()



### TODO: Compile the model.
Xception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])



### TODO: Train the model.
# checkpointer_Xception = ModelCheckpoint(filepath=r'D:\Courses\Udacity Nanodegrees\Machine Learning Nanodegree\Projects\Project 3 - Dog Breed Classifier\dog_breed_classifier\saved_models\weights_best_Xception.hdf5', verbose=1, save_best_only=True)
# Xception_model.fit(train_Xception, train_targets, validation_data=(valid_Xception, valid_targets),
         # callbacks=[checkpointer_Xception], nb_epoch=25, batch_size = 40, verbose = 1)


# ### (IMPLEMENTATION) Load the Model with the Best Validation Loss

# In[29]:


### TODO: Load the model weights with the best validation loss.
Xception_model.load_weights(r'D:\Courses\Udacity Nanodegrees\Machine Learning Nanodegree\Projects\Project 3 - Dog Breed Classifier\dog_breed_classifier\saved_models\weights_best_Xception.hdf5')

# get index of predicted dog breed for each image in test set
Xception_predictions = [np.argmax(Xception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Xception]

# report test accuracy
# test_accuracy = 100*np.sum(np.array(Xception_predictions)==np.argmax(test_targets, axis=1))/len(Xception_predictions)
# print('Test accuracy: %.4f%%' % test_accuracy)


def Xception_predict_breed(img_path):
    # extract bottleneck features
#     print('image shape: ', img_path)
    image_tensor = path_to_tensor(img_path)
#     print('tensor shape: ', image_tensor.shape)
    image_bottleneck_feature = extract_Xception(image_tensor)
#     print('bottleneck features shape: ', image_bottleneck_feature.shape)
    # obtain predicted vector
    predicted_vector_Xception = Xception_model.predict(image_bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector_Xception)]


# a function that checks if the image contains a face or a dog and returns the corresponding string to be printed
# import matplotlib.image as mpimg
# def dog_or_human(filepath):
#     dog = dog_detector(filepath)
#     human = face_detector(filepath)
#     return dog, human

# def classify_dog_breed(filepath):
#     # img = mpimg.imread(filepath)
#     # fig = plt.figure()
#     # ax = fig.add_subplot(1,1,1)
#     # ax.imshow(img) #show image
#     img = cv2.imread(filepath)
#     # cv2.imshow('img', img)
#     # dog, human = dog_or_human(filepath)
#     if(dog):
#         breed = Xception_predict_breed(filepath)
#         print('You look like a', breed)
#     elif(human):
#         breed = Xception_predict_breed(filepath)
#         print("If you were a dog you'd be a", breed)
#     else:
#         print("This is neither a dog nor a valid human face!")

def classify_dog_breed(filepath):
    # img = mpimg.imread(filepath)
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.imshow(img) #show image
    img = cv2.imread(filepath)
    # cv2.imshow('img', img)
    # dog, human = dog_or_human(filepath)
    breed = Xception_predict_breed(filepath)
    print('You look like a', breed)



classify_dog_breed(r'D:\Courses\Udacity Nanodegrees\Machine Learning Nanodegree\Projects\Project 3 - Dog Breed Classifier\dog_breed_classifier\testing_files\me.jpg')
cv2.waitKey(0)
