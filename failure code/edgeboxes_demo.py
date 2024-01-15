# 窗口式EdgeBoxes调参
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

def on_trackbar_change(_):
    alpha = cv2.getTrackbarPos('alpha', 'EdgeBoxes') / 100.0
    beta = cv2.getTrackbarPos('beta', 'EdgeBoxes') / 100.0
    eta = cv2.getTrackbarPos('eta', 'EdgeBoxes') / 100.0
    minScore = cv2.getTrackbarPos('minScore', 'EdgeBoxes') / 100.0
    maxBoxes = cv2.getTrackbarPos('maxBoxes', 'EdgeBoxes')
    edgeMinMag = cv2.getTrackbarPos('edgeMinMag', 'EdgeBoxes') / 100.0
    edgeMergeThr = cv2.getTrackbarPos('edgeMergeThr', 'EdgeBoxes') / 100.0
    clusterMinMag = cv2.getTrackbarPos('clusterMinMag', 'EdgeBoxes') / 100.0
    maxAspectRatio = cv2.getTrackbarPos('maxAspectRatio', 'EdgeBoxes') / 10.0
    minBoxArea = cv2.getTrackbarPos('minBoxArea', 'EdgeBoxes') * 1000
    gamma = cv2.getTrackbarPos('gamma', 'EdgeBoxes') / 10.0
    kappa = cv2.getTrackbarPos('kappa', 'EdgeBoxes') / 10.0

    # Update EdgeBoxes with new parameters
    edge_boxes.setAlpha(alpha)
    edge_boxes.setBeta(beta)
    edge_boxes.setEta(eta)
    edge_boxes.setMinScore(minScore)
    edge_boxes.setMaxBoxes(maxBoxes)
    edge_boxes.setEdgeMinMag(edgeMinMag)
    edge_boxes.setEdgeMergeThr(edgeMergeThr)
    edge_boxes.setClusterMinMag(clusterMinMag)
    edge_boxes.setMaxAspectRatio(maxAspectRatio)
    edge_boxes.setMinBoxArea(minBoxArea)
    edge_boxes.setGamma(gamma)
    edge_boxes.setKappa(kappa)

    # Detect boxes and display the results
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)
    display_image = image.copy()
    # 在图像上绘制提议框
    for i, b in enumerate(boxes):
        x, y, w, h = b
        cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
        # # 保存图片
        # crop_img = image[y:y+h, x:x+w]
        # cv2.imwrite(f'../output/crop_img_{i}.jpg', crop_img)

    cv2.imshow('EdgeBoxes', display_image)

if __name__ == '__main__':
    # 读取图像
    image = cv2.imread("../test_images/1.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 图像预处理
    # image = gamma_correction(image, gamma=2)

    # 创建EdgeBoxes
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection("model.yml.gz")
    rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv2.ximgproc.createEdgeBoxes()

    # 初始参数设置
    edge_boxes.setMaxBoxes(30)  # 设置返回的最大边界框数
    # edge_boxes.setAlpha(0.65)  # 设置步长控制参数
    edge_boxes.setBeta(0.75)  # 设置群组边缘强度控制参数
    edge_boxes.setMinScore(0.1)  # 设置检测到的边界框的最小得分
    edge_boxes.setMaxAspectRatio(5.0)  # 设置边界框的最大长宽比
    edge_boxes.setMinBoxArea(50000)  # 设置边界框的最小面积
    edge_boxes.setGamma(2.0)        # 设置群组边缘的聚合程度
    edge_boxes.setKappa(1.5)        # 设置用于评分的聚合边缘强度的控制参数

    # 获取提议框
    boxes, _ = edge_boxes.getBoundingBoxes(edges, orimap)

    # Create window with trackbars
    cv2.namedWindow('EdgeBoxes')
    cv2.createTrackbar('alpha', 'EdgeBoxes', 65, 100, on_trackbar_change)
    cv2.createTrackbar('beta', 'EdgeBoxes', 75, 100, on_trackbar_change)
    cv2.createTrackbar('eta', 'EdgeBoxes', 100, 100, on_trackbar_change)
    cv2.createTrackbar('minScore', 'EdgeBoxes', 10, 100, on_trackbar_change)
    cv2.createTrackbar('maxBoxes', 'EdgeBoxes', 30, 100, on_trackbar_change)
    cv2.createTrackbar('edgeMinMag', 'EdgeBoxes', 10, 100, on_trackbar_change)
    cv2.createTrackbar('edgeMergeThr', 'EdgeBoxes', 50, 100, on_trackbar_change)
    cv2.createTrackbar('clusterMinMag', 'EdgeBoxes', 50, 100, on_trackbar_change)
    cv2.createTrackbar('maxAspectRatio', 'EdgeBoxes', 30, 100, on_trackbar_change)
    cv2.createTrackbar('minBoxArea', 'EdgeBoxes', 10, 100, on_trackbar_change)
    cv2.createTrackbar('gamma', 'EdgeBoxes', 20, 100, on_trackbar_change)
    cv2.createTrackbar('kappa', 'EdgeBoxes', 15, 100, on_trackbar_change)
   
    # Initial call to update display
    on_trackbar_change(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()