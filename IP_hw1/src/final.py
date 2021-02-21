from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os
import time

def genMatrix(file):
    return [[int(val) for val in line.split(',')] for line in file]

def getParams(argss):
    se = argv[2]
    input_File = argv[3]
    output_File = argv[4]

    if len(argss) < 5 or argv[1] not in ('e', 'd', 'o', 'c'):
        print("usage: e/d/o/c <SE file> <input file> <output file>")
        quit()

    if argv[1] == 'e':
        argument = 1
    elif argv[1] == 'd':
        argument = 2
    elif argv[1] == 'o':
        argument = 3
    elif argv[1] == 'c':
        argument = 4

    return argument, se, input_File, output_File

def erosion(matrix, se):
    res = np.array(matrix)                          #Takes BINARY image, value of output pixel is minimum value of all
    mini_val = np.max(res)                          #the pixels in the input pixels neighbourhood. Finds smallest element in sub-matrix
    ans = np.zeros((len(matrix), len(matrix[0])))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            ans[i][j] = min_val(matrix, i, j, se, mini_val)
    return ans


def min_val(matrix, a, b, se, mini_val):
    j = b
    for x in range(len(se)):
        for y in range(len(se[0])):
            if se[x][y] == 1:
                if 0 <= b < len(matrix[0]):
                    if matrix[a][b] < mini_val:
                        mini_val = matrix[a][b]
                    b += 1
                else:
                    break
        if 0 <= a < len(matrix) - 1:
            a += 1
            b = j
        else:
            break
    return mini_val


def dilation(matrix, se):
    res = np.array(matrix)
    maxi = np.min(res)
    ans = np.zeros((len(matrix), len(matrix[0])))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            ans[i][j] = max_val(matrix, i, j, se, maxi)
    return ans


def max_val(matrix, a, b, se, maxi_val):
    i = a
    j = b
    for x in range(len(se)):
        for y in range(len(se[0])):
            if se[x][y] == 1:
                if 0 <= b < len(matrix[0]):
                    if matrix[a][b] >= maxi_val:
                        maxi_val = matrix[a][b]
                    b += 1
                else:
                    break
        if 0 <= a < len(matrix) - 1:
            a += 1
            b = j
        else:
            break
    return maxi_val


def opening(matrix, se):
    ans = np.zeros((len(matrix), len(matrix[0]))) #WE DO EROSION AND AFTER WE DO DILATION
    ans = erosion(matrix, se)
    return dilation(ans, se)


def closing(matrix, se):
    ans = np.zeros((len(matrix), len(matrix[0]))) #WE DO DILATION AND AFTER WE DO EROSION
    ans = dilation(matrix, se)
    return erosion(ans, se)


def main():
    input_arg, s, input_File, output_File = getParams(argv)
    m_file = argumenten("input/" + input_File, "r", errors="ignore")
    matrix = genMatrix(m_file)
    se = argumenten("input/" + s, "r", errors="ignore")
    se_matrix = genMatrix(se)
   
    if input_arg == 1:
        np.savetxt("output/experiment/" + output_File, erosion(matrix, se_matrix), fmt='%i', delimiter=',')
        cv2.imwrite("output/" + output_File+".png", np.array(erosion(matrix, se_matrix)))
    elif input_arg == 2:
        np.savetxt("output/" + output_File, dilation(matrix, se_matrix), fmt='%i', delimiter=',')
        cv2.imwrite("output/" + output_File+".png", np.array(dilation(matrix, se_matrix)))
    elif input_arg == 3:
        np.savetxt("output/" + output_File, opening(matrix, se_matrix), fmt='%i', delimiter=',')
        cv2.imwrite("output/" + output_File+".png", np.array(opening(matrix, se_matrix)))
    elif input_arg == 4:
        np.savetxt("output/" + output_File, closing(matrix, se_matrix), fmt='%i', delimiter=',')
        cv2.imwrite("output/" + output_File+".png", np.array(closing(matrix, se_matrix)))


main()
