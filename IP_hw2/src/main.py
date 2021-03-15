from sys import argv
import numpy as np
from _collections import deque
import cv2
from PIL import Image
import imageio
from scipy import ndimage

MASK = -2
WSHD = 0
INIT = -1
INQUEUE = -3
LVLS = 256

def watershed(img, n):
    que = deque()
    try:
        height, width = img.shape
        total = height * width
    except:
        try:
            img.transpose(2,0,1).reshape(3,-1)
            height, width = img.shape
            total = height * width
        except:
            print("This file is not supported")
            exit()

    # output image matrix, initialized with INIT
    label = np.full((height, width), INIT, np.int32) 

    # Flattening the image
    flat_img = img.reshape(total)
    
    # Getting Y and X of the picture
    pixels = getPixels(height, width) 
    print("Getting {} Neighbours...".format(n))
    
    # Getting Y and X of the neighbour of image
    neighbours = np.array([getNeighbors(height, width, p, n) for p in pixels])
    neighbours = neighbours.reshape(height, width)

    current = 0
    block = False
    
    # Sorting pixels 
    idx = np.argsort(flat_img)
    sorted_image = flat_img[idx]
    sorted_pixels = pixels[idx]

    # Creating 256 grey value levels
    lvls = np.linspace(sorted_image[0], sorted_image[-1], LVLS)
    lvl_idx = []
    current_lvl = 0

    # checking for different gray level indices
    for i in range(total):
        if sorted_image[i] > lvls[current_lvl]:
            while sorted_image[i] > lvls[current_lvl]: 
                current_lvl += 1
            lvl_idx.append(i)
    lvl_idx.append(total)
    
    start = 0

    for stop in lvl_idx:

        # Masking all pixels at current level
        for pixel in sorted_pixels[start:stop]:
            label[pixel[0], pixel[1]] = MASK
            # Adding the neighbours of existing basins to queue
            for nbr_pixel in neighbours[pixel[0], pixel[1]]:
                if label[nbr_pixel[0],nbr_pixel[1]] >= WSHD:
                    label[pixel[0], pixel[1]] = INQUEUE
                    que.append(pixel)
                    break

        while que:
            pixel = que.popleft()
            for nbr_pixel in neighbours[pixel[0], pixel[1]]:
                pixel_label = label[pixel[0], pixel[1]]
                nbr_label = label[nbr_pixel[0],nbr_pixel[1]]
                if nbr_label > 0:
                    if pixel_label == INQUEUE or (pixel_label == WSHD and block):
                        label[pixel[0], pixel[1]] = nbr_label
                    elif pixel_label > 0 and pixel_label != nbr_label:
                        label[pixel[0], pixel[1]] = WSHD
                        block = False
                elif nbr_label == WSHD:
                    if pixel_label == INQUEUE:
                        label[pixel[0], pixel[1]] = WSHD
                        block = True
                elif nbr_label == MASK:
                    label[nbr_pixel[0], nbr_pixel[1]] = INQUEUE
                    que.append(nbr_pixel)

        # Looking for new minimas
        for pixel in sorted_pixels[start:stop]:
            if label[pixel[0], pixel[1]] == MASK:
               current += 1
               que.append(pixel)
               label[pixel[0], pixel[1]] = current
               while que:
                  q = que.popleft()
                  for r in neighbours[q[0], q[1]]:
                     if label[r[0], r[1]] == MASK:
                        que.append(r)
                        label[r[0], r[1]] = current

        start = stop 

    return label

def getPixels(height, width):
    y_axis = np.zeros((height, width))
    for j in range(height):
        row = np.full((1, width), j)
        y_axis[j] = row

    x_axis = np.zeros((height, width))
    try:
        row = np.arange(0, height) 
        for i in range(height):
            x_axis[i] = row
    except:
        try:
            row = np.arange(0, height + 1) 
            for i in range(height):
                x_axis[i] = row
        except:
            row = np.arange(0, width)
            for i in range(height):
                x_axis[i] = row
        
    plane = np.zeros((2, height, width))
    plane[0] = y_axis
    plane[1] = x_axis

    return plane.reshape(2, -1).T.astype(int)

def getNeighbors(height, width, pixel, n):
    i = max(0, pixel[0] - 1)
    j = min(height, pixel[0] + int(n / 4))

    x = max(0, pixel[1] - 1)
    y = min(width, pixel[1] + int(n / 4))

    matrix = np.zeros(shape=(j - i, y - x))
    for t in range(j - i):
        temp = np.full((1, (y - x)), t + i)
        matrix[t] = temp

    temp = np.arange(x, y)
    matrix2 = np.zeros(shape=(j - i, y - x))
    for t in range(y - x):
       if t == j - i: 
           break
       matrix2[t] = temp
    glob = np.zeros((2, j - i, y - x))
    glob[0] = matrix
    glob[1] = matrix2

    return glob.reshape(2, -1).T.astype(int)

def getParams(argss):
    end = False
    if argv[3].endswith(".txt"):end = True

    distanced = argv[1]
    neighbours = argv[2]
    infile = "../input/" + argv[3]
    outfile = "../output/" + argv[4]

    return end, distanced, int(neighbours), infile, outfile

def genMatrix(filename):
    with open(filename, 'r') as f: mat = [[int(num) for num in line.split(',')] for line in f]
    imageio.imwrite('../output/genmatrix/matrix.png', mat)
    return mat

def distanceTransform(filename):
    with open(filename, 'r') as f: l = [[int(num) for num in line.split(',')] for line in f]
    img = ndimage.distance_transform_edt(l)
    imageio.imwrite('../output/distanced/distanced.png', img)
    return img

def main():
    t, d, n, infile, outfile = getParams(argv)
    if t:
        if d == "yes":
            print("Textfile, Distance Transform")
            distanceTransform(infile)
            img = np.array(Image.open("../output/distanced/distanced.png"))
            imageio.imwrite(outfile, watershed(img, n))
        elif d == "no":
            print("Textfile, No Distance Transform")
            genMatrix(infile)
            img = np.array(Image.open("../output/genmatrix/matrix.png"))
            imageio.imwrite(outfile, watershed(img, n))
        else:
            print("Textfile, Just Distance Transform")
            distanceTransform(infile)
    else:
        img = np.array(Image.open(infile))
        if (len(img.shape) > 2):
            print("Image, Converting to Grayscale")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print("Gray Scale Image")
        img = cv2.medianBlur(img, 5)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        imageio.imwrite(outfile, watershed(img, n))


if __name__ == "__main__":
    main()