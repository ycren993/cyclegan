import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 定义Sobel算子
class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        # 水平Sobel滤波器
        self.sobel_filter_x = torch.tensor([[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]], dtype=torch.float)
        # 垂直Sobel滤波器
        self.sobel_filter_y = torch.tensor([[-1, -2, -1],
                                            [0, 0, 0],
                                            [1, 2, 1]], dtype=torch.float)

    def forward(self, x):
        # 将滤波器转换为可以广播的形状
        sobel_filter_x = self.sobel_filter_x.view((1, 1, 3, 3))
        sobel_filter_y = self.sobel_filter_y.view((1, 1, 3, 3))

        # 应用Sobel算子进行边缘检测
        sobel_x = F.conv2d(x, sobel_filter_x, bias=None, stride=1, padding=1, groups=1)
        sobel_y = F.conv2d(x, sobel_filter_y, bias=None, stride=1, padding=1, groups=1)

        # 计算Sobel算子的绝对值
        sobel = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)

        return sobel


# 读取图片并转换为PyTorch张量
image = plt.imread('/home/ycren/python/testpic/Snipaste_2024-12-29_17-41-06.png')  # 替换为你的图片路径
image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0  # 转换为PyTorch张量，归一化到[0,1]范围

# 实例化Sobel类
sobel_operator = Sobel()

# 应用Sobel算子
edges = sobel_operator(image.unsqueeze(0))  # 添加一个维度以匹配PyTorch的batch维度

# 显示原图和边缘检测后的图片
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image.permute((1, 2, 0)))
ax1.set_title('Original Image')
ax2.imshow(edges.squeeze(0).permute((1, 2, 0)))
ax2.set_title('Edges')
plt.show()