# 去除阴影尝试
import numpy as np
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("../test_images/1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 光滑
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# 二值化
binary = cv2.adaptiveThreshold(gray, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 91, 6)

# 除噪点
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel1, iterations=3)

# 扩张白色区域
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel2, iterations=5)

plt.figure(1)
plt.imshow(close, cmap='gray')
plt.show()

