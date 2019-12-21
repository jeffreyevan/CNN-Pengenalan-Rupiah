import numpy as np
import cv2

image = cv2.imread('1000_001.jpg')
resized_image = cv2.resize(image, (500, 500)) 
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
y = 200
x = 0
h = 150
w = 200
crop = gray_image[y:y+h, x:x+w]
cv2.imwrite("1000-1.jpg", crop)