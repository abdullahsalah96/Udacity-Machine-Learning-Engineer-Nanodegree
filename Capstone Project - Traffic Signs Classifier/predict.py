from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, GlobalAveragePooling2D, Conv2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
# from train import loaded_model
from train import test_tensors, test_labels
from utils import load_images, path_to_tensor, paths_to_tensor
import numpy as np

def predict_class(model, img_path):
    """ 
    A funtion that takes the model and the path of the image to be predicted and returns the prediction
    """
    prediction = model.predict(path_to_tensor(img_path))
    return(np.argmax(prediction))

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])


print(predict_class(loaded_model, r"D:\Courses\Udacity Nanodegrees\Machine Learning Nanodegree\Capstone Project\Traffic signs dataset\BelgiumTSC_Testing\Testing\00000\00017_00002.ppm"))
