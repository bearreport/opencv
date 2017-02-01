import os, sys

import cv2
import numpy as np
from PIL import Image

# Select an image and a cascade - store in root folder for easy use
imagePath = 'fam.jpg'
cascFacePath = 'haarcascade_frontalface_default.xml'
cascEyesPath = 'haarcascade_eye.xml'

#convert from xml to cv cascade class
faceCascade = cv2.CascadeClassifier(cascFacePath)
eyesCascade = cv2.CascadeClassifier(cascEyesPath)

# Read the image and convert to grayscale for easier face finding
image = cv2.imread(imagePath)
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces using the default "face finding settings"
faces = faceCascade.detectMultiScale(
    grayscale,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

eyes = eyesCascade.detectMultiScale(
    grayscale,
    scaleFactor=1.1,
    minNeighbors=2,
    minSize=(15, 15)
)


print("Found {0} faces!".format(len(faces)))
print("Found {0} eyes!".format(len(eyes)))

# Draw Rectangles Around The Faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)

# Draw Rectangles Around The Eys
for (x, y, w, h) in eyes:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("Faces found", image)
cv2.imwrite("foundfaces.jpeg", image)
cv2.waitKey(0)
