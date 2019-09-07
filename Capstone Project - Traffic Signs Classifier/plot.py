import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread(r"D:\Courses\Udacity Nanodegrees\Machine Learning Nanodegree\Capstone Project\Traffic signs dataset\BelgiumTSC_Training\Training\00000\01797_00000.ppm")
color = ('b','g','r')

for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.show()

cv2.imshow("image", img)
cv2.waitKey(0)
