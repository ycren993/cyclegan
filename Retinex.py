import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def retinex_single_scale(img, sigma=80):
    """
    单尺度Retinex图像增强算法
    :param img: 输入图像（BGR格式）
    :param sigma: 高斯核标准差
    :return: 增强后的图像（RGB格式）
    """
    # 转换颜色空间并归一化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img.astype(np.float32) / 255.0

    # 估计光照分量（高斯模糊）
    illumination = cv2.GaussianBlur(img_float, (0, 0), sigma)

    # 防止除零错误
    np.maximum(illumination, 1e-6, illumination)

    # 计算反射分量（Retinex核心公式）
    reflection = np.log10(img_float + 1e-6) - np.log10(illumination)

    # 对每个通道进行归一化处理
    for ch in range(3):
        min_val = reflection[:, :, ch].min()
        max_val = reflection[:, :, ch].max()
        if max_val > min_val:
            reflection[:, :, ch] = (reflection[:, :, ch] - min_val) / (max_val - min_val)

    return (np.clip(reflection, 0, 1) * 255).astype(np.uint8)

# 读取图像
input_img = cv2.imread('/home/ycren/shells/result_test_test/testparam/test_latest/images/000657_jpg.rf.6e205aa3e96ba75cb2ba27c46d3caffa_fake.png')  # 请替换为你的图片路径

# 应用Retinex算法
output_img = retinex_single_scale(input_img, sigma=80)

# 使用matplotlib展示结果
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
print(torch.from_numpy(output_img).permute(2,0,1).float()/255)
plt.imshow(output_img)
plt.title('Retinex Enhanced')
plt.axis('off')

plt.tight_layout()
plt.show()

