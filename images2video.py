import cv2
import numpy as np
import glob
 
img_array = []

# Folder where the images are stored
nameFile = input("Enter path: ")
for filename in glob.glob( nameFile + '/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('original.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()