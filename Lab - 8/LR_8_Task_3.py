import cv2
import numpy as np

img = cv2.imread("Liashuk2.png")
print(img.shape)

imgResize = cv2.resize(img, (1000, 500))
print(imgResize.shape)

imgCropped = img[40:420, 90:380]

cv2.imshow("Image Cropped", imgCropped)

cv2.waitKey(0)