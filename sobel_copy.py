

import torch
import torch.nn as nn
import cv2

class SobelConv(nn.Module):
    def __init__(self):
        super(SobelConv, self).__init__()
        # 构建Sobel滤波器
        self.sobel_filter = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)
        # 初始化Sobel滤波器的权重
        sobel_kernel = torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],  # X方向
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  # Y方向
        ])
        self.sobel_filter.weight.data = sobel_kernel.view(2, 3, 3, 3).permute(0, 3, 1, 2).float()

    def forward(self, x):
        # 应用Sobel滤波器
        sobel = self.sobel_filter(x)
        # 计算边缘强度
        edge_map = torch.sqrt(torch.pow(sobel[0], 2) + torch.pow(sobel[1], 2))
        return edge_map
image = cv2.imread('/home/ycren/python/testpic/Snipaste_2024-12-29_17-41-06.png', cv2.IMREAD_GRAYSCALE)

image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

# 示例用法
input_image = torch.randn(1, 3, 240, 320)  # 假设输入图像大小为240x320
model = SobelConv()
edges = model(image_tensor)

# 显示边缘图
# 需要注意的是，这里仅用于展示，实际情况下你可能需要使用图像处理库如PIL/Pillow来处理和显示图像
import matplotlib.pyplot as plt

plt.imshow(edges[0].numpy(), cmap='gray')
plt.show()