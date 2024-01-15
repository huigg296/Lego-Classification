import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
color_image = cv2.imread("../test_images/3.jpg")
image = cv2.cvtColor(color_image,  cv2.COLOR_BGR2GRAY)

# Preprocess: smooth the image
image = cv2.GaussianBlur(image, (5,5), 0)

# Thresholding with OTSU
binary = cv2.adaptiveThreshold(image, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 91, 6)
cv2.imshow('binary', binary)

# Obtain sure background area
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)
sure_bg = cv2.dilate(opening, kernel, iterations=5)
cv2.imshow('sure_bg', sure_bg)

# Obtain sure foregroud area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255, cv2.THRESH_BINARY)
sure_fg = np.uint8(sure_fg)
cv2.imshow('sure_fg', sure_fg)

# Obtain unknown area
unknown = cv2.subtract(sure_bg, sure_fg)
cv2.imshow('unknown', unknown)

# Make markers
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1       # Mark the sure background with 1
markers[unknown==255] = 0   # Mark the unknown area with 0

# Watershed
markers = cv2.watershed(color_image, markers)

# Fill the result image based on the markers
result = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
result[markers == -1] = [255, 0, 0]
result[markers == 1] = [255, 255, 255]
cv2.imshow('result', result)

# Display
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('../images/watershed-binary.jpeg', binary)
cv2.imwrite('../images/watershed-opening.jpeg', opening)
cv2.imwrite('../images/watershed-sure-bg.jpeg', sure_bg)
cv2.imwrite('../images/watershed-sure-fg.jpeg', sure_fg)
cv2.imwrite('../images/watershed-unknown.jpeg', unknown)
cv2.imwrite('../images/watershed-result.jpeg', result)