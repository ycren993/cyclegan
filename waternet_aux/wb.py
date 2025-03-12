import cv2
import numpy as np
import matplotlib.pyplot as plt


def white_balance(image):
    # 计算每个通道的平均值
    avg_per_channel = cv2.mean(image)[:3]

    # 计算每个通道的目标值
    target = 255
    scale = target / np.array(avg_per_channel)

    # 按比例调整通道
    img_result = image * scale
    img_result = np.clip(img_result, 0, 255).astype(np.uint8)

    return img_result


# 读取图像
img = cv2.imread('/home/ycren/python/cycle_test/testA/test_9001up.jpg')
wb_img = white_balance(img)

# 显示结果
plt.subplot(1, 2, 1)
plt.title('原始图像')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('经过白平衡处理的图像')
plt.imshow(cv2.cvtColor(wb_img, cv2.COLOR_BGR2RGB))

plt.show()