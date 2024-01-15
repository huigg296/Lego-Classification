# Edgeboxes尝试
import numpy as np
from matplotlib import pyplot as plt
import cv2

def gamma_correction(image, gamma=1.0):
    # 构建查找表，实现伽马校正
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # 应用伽马校正
    return cv2.LUT(image, table)

# Load your image
image = cv2.imread("../test_images/1.jpg")

# 图像预处理
# image = cv2.GaussianBlur(image, (7, 7), 0)
image = gamma_correction(image, gamma=1.5)

# 创建EdgeBoxes
edge_detection = cv2.ximgproc.createStructuredEdgeDetection("model.yml.gz")
rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

orimap = edge_detection.computeOrientation(edges)
edges = edge_detection.edgesNms(edges, orimap)

edge_boxes = cv2.ximgproc.createEdgeBoxes()

# 初始参数设置
edge_boxes.setMaxBoxes(30)  # 设置返回的最大边界框数
edge_boxes.setAlpha(0.3)  # 设置步长控制参数
edge_boxes.setBeta(0.01)  # 设置群组边缘强度控制参数
edge_boxes.setMinScore(0.01)  # 设置检测到的边界框的最小得分
edge_boxes.setMaxAspectRatio(5.0)  # 设置边界框的最大长宽比
edge_boxes.setMinBoxArea(50000)  # 设置边界框的最小面积
edge_boxes.setGamma(2.0)        # 设置群组边缘的聚合程度
edge_boxes.setKappa(1.5)        # 设置用于评分的聚合边缘强度的控制参数

# 获取提议框
boxes, scores = edge_boxes.getBoundingBoxes(edges, orimap)

# Draw boxes on the image
for i, b in enumerate(boxes):
    x, y, w, h = b
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2, cv2.LINE_AA)

    crop_img = image[y:y+h, x:x+w]
    cv2.imwrite(f'../output/crop_img_{i}.jpg', crop_img)

# Show the image with boxes
plt.figure(1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.figure(2)
plt.imshow(edges)
plt.show()
