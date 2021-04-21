
import cv2
import numpy as np
from skimage.util import random_noise
from sys import argv
 
img = argv[1]
filterAmount = 0.1

# Load the image
img = cv2.imread("C:/Users/shahi/Desktop/SEM6/ImgProcessing3/input/" + img + ".png")
 
# Add salt-and-pepper noise to the image.
noise_img = random_noise(img, mode='s&p',amount=0.3)
 
# The above function returns a floating-point image
# on the range [0, 1], thus we changed it to 'uint8'
# and from [0,255]
noise_img = np.array(255*noise_img, dtype = 'uint8')
 
# Display the noise image

cv2.imshow('blur',noise_img)
cv2.waitKey(0)
