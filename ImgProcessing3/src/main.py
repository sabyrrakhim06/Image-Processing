#----------------------Mutual Information------------------------------
#--------------Shahin Mammadov, Abumansur Sabyrrakhim------------------
#------------Execution: python main.py "image" "bin size"--------------
#-------------------------21/04/2021-----------------------------------


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sys import argv
import seaborn as sns

def GetComm(argss):
    return argv[1], wargv[2]

class MutualInformation:
    
    def __init__(self, image):
        self.image = image
        self.h = self.getShape(image)[0]
        self.w = self.getShape(image)[1]
    
    def getShape(self,image):
        return np.shape(image)
        
    def getArray(self, image):
        image1D = np.ravel(image)
        return image1D
        
    def setBinSize(self, size):
        self.bin_size = size

    def coloredChannels(self, blue, green, red):
        zeros = np.zeros(blue.shape, np.uint8)

        blueBGR = cv2.merge((blue,zeros,zeros))
        greenBGR = cv2.merge((zeros,green,zeros))
        redBGR = cv2.merge((zeros,zeros,red))

        return blueBGR,greenBGR,redBGR
        
    def splitChannels(self):
        blue,green,red = cv2.split(self.image)
        return  blue,green,red
    
    def cropImage(self, image, x, x1 ,y, y1, h, w):
        cropedImg = image[y : y1 + h, x : x1 + w].copy() #cropping the image
        return cropedImg

    def getEntropy(self, hist):
        dataNormalized = hist[0] / float(np.sum(hist[0]))
        noneZeroData= dataNormalized[dataNormalized != 0]
        ent = -(noneZeroData*np.log(np.abs(noneZeroData))).sum() 
        return ent
    
    # method for finding mutual information
    def mutualInformation(self, img1, img2):
        HistX = np.histogram(img1, bins= self.bin_size, range=(0, 256), density=True)
        HistY = np.histogram(img2, bins= self.bin_size, range=(0, 256), density=True)
        HistXY = np.histogram2d(img1,img2, bins= self.bin_size)

        entX = self.getEntropy(HistX)
        entY = self.getEntropy(HistY)
        jointEntXY = self.getEntropy(HistXY)
    
        return (entX + entY - jointEntXY)

    def getMIPlot(self, x, y, fileName):
        plt.ylabel('Mutual Information')
        plt.xlabel('Image Translations')
        plt.plot(x, y)
        plt.savefig('../output/' + fileName + '_MI_plot.pdf') #mutual information x image translations plot
    


userInput, binSize = GetComm(argv)  #declaring the image and bin size from command line 

image = cv2.imread('../input/' + userInput + '.png',1)

execute = MutualInformation(image) #create object

b, g, r = execute.splitChannels()

green = execute.cropImage(g, 20, 20, 0, 0, execute.h, execute.w) # cutting 20 pixels from left and right
green_h, green_w = execute.getShape(green)

green = execute.cropImage(green, 0, -20, 0, 0, green_h, green_w)
green_h, green_w = execute.getShape(green)
green_size = green_h * green_w


green_Array = execute.getArray(green)
execute.setBinSize(binSize) #setting the bin size

mi_array = []
directionX = 0
while directionX != 41:
    red_pixels = (np.mgrid[0:green_h, directionX: directionX + green_w]).reshape(2, -1).T
    red_Array = []
    for x in red_pixels:
        red_Array.append(r[x[0]][x[1]])  #virtually moving the red channel images in x-direction over the corresponding green channel image in 41 steps from left to right

    red_Array = execute.getArray(red_Array)

    mi_res = execute.mutualInformation(green_Array,red_Array)
    mi_array.append(mi_res)

    directionX = directionX + 1

arr_iterations = np.arange(0, 41)

execute.getMIPlot(arr_iterations, mi_array,userInput)

cv2.imwrite('../output/' + userInput + '_green.png', g) #saving the output
cv2.imwrite('../output/' + userInput + '_red.png', r)
cv2.imwrite('../output/' + userInput + '_green_cropped.png', green)
