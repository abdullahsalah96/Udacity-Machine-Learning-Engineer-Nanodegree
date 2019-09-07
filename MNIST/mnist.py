from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile
from PIL import Image
import cv2

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    # img = image.load_img(color_to_grayscale(img_path), target_size=(28, 28))
    img = image.load_img(color_to_grayscale(img_path), target_size=(28, 28))
    # convert PIL.Image.Image type to 3D tensor with shape (28, 28, 3)
    x = image.img_to_array(img)
    print('img shape: ', x.shape[:])
    # convert 3D tensor to 4D tensor with shape (1, 28, 28, 1) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

#loading training and testing data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('Number of training images: ', X_train.shape[0])
print('Number of testing images: ', X_test.shape[0])
print('Dimensions of images: ', X_train.shape[1:])
#visualizing some images
figure = plt.figure(figsize=(28,28))
for i in range(6):
    subplot = figure.add_subplot(1, 6, i+1) #create a subplot to show 6 training images
    subplot.imshow(X_train[i], cmap='gray')
    subplot.set_title(str(y_train[i])) #show the label of each image


#normalizing the features so that each pixel has avalue between 0 and 1 instead of 0-255
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
#one hot encoding the labels
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print('One hot encoded feature: ', y_train[1])

#defining our model

model = Sequential() #creating our Sequential object
model.add(Flatten(input_shape = X_train.shape[1:])) #Flattening our features to be fed into the Neural network as input
model.add(Dense(256, activation = 'relu')) #adding a 256 node layer
model.add(Dropout(0.2)) #Making each node has a 0.2 probability of being dropped out
model.add(Dense(512, activation = 'relu')) #adding a 512 node layer
model.add(Dropout(0.3)) #Making each node has a 0.2 probability of being dropped out
model.add(Dense(256, activation = 'relu')) #adding a 256 node layer
model.add(Dropout(0.4)) #Making each node has a 0.2 probability of being dropped out
model.add(Dense(10, activation = 'softmax')) #adding an output 10 nodes' layer
model.summary() #printing a summary of the model

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy']) #compiling our model


''' THE FOLLOWING STEPS ARE USED ONLY WHEN TRAINING OUR MODEL'''
# making a checkpointer for our model to save the weights -- used only when training our model
# checkpointer = ModelCheckpoint(filepath = 'D:/Courses/Udacity Nanodegrees/Machine Learning Nanodegree/self_projects/Text recognition using keras/mnist_best_weights.hdf5', verbose = 1, save_best_only = True)

#fitting the model and passing the checkpointer to it -- Used only when training our model
# model.fit(X_train, y_train, batch_size = 200, nb_epoch = 20, validation_split=0.2, callbacks=[checkpointer], shuffle = True)

model.load_weights('D:/Courses/Udacity Nanodegrees/Machine Learning Nanodegree/self_projects/Text recognition using keras/mnist_best_weights.hdf5')
#evaluating the performance of our model
score = model.evaluate(X_test, y_test, verbose = 1)
accuracy = score[1] *100 #score[0] returns loss value, score[1] returns the metrics value (accuracy)
print(r'\n\nAccuracy score: ', accuracy)

cap = cv2.VideoCapture()
cap.open(0)
while True:
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.imread('D:/Courses/Udacity Nanodegrees/Machine Learning Nanodegree/self_projects/Text recognition using keras/1.png', cv2.IMREAD_GRAYSCALE) #load image as grayscale
    # img2 = cv2.bitwise_not(img) #invert the image to match the mnist dataset
    ret,thresh = cv2.threshold(img,100,200,cv2.THRESH_BINARY_INV)
    cv2.imshow('thresh', thresh)

    img2 = np.array(thresh)
    img2 = cv2.resize(img2, (28,28)) #resize the image to be the match the input of our model

    img2 = img2.astype('float32')/255 #normalizing the image
    # print('img matrix: ', img)
    prediction = model.predict(img2.reshape(1, 28, 28)) #Get the prediction array of the model after reshaping the image
    print('Prediction matrix: ', prediction)
    prediction = np.argmax(prediction)  #get the index where the probability is maximum
    print('Predicted number: ', prediction)
    cv2.putText(img, str(prediction), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
    cv2.imshow('img', img)
    if (cv2.waitKey(1)==27):
        break
