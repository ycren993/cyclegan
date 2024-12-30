import cv2
import numpy as np
import matplotlib.pyplot as plt
def Prewitt(srcImg_path):
    img = cv2.imread(srcImg_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Prewitt算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    # 转成uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # 显示图形
    titles = ["Original Image", "Prewitt Image"]
    images = [img, Prewitt]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.axis('off')
    plt.show()
if __name__ == '__main__':
    Prewitt('/home/ycren/python/testpic/Snipaste_2024-12-29_17-41-06.png')