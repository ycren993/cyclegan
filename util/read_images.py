import os

import torch
import torch.nn.functional as F


def prewitt_operator(image):
    # Define Prewitt kernels
    '''
    kernel_x = torch.tensor([[[[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]],

                              [[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]],

                              [[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]]
                              ]], dtype=torch.float32, device='cuda')

    kernel_y = torch.tensor([[
        [[1, 1, 1],
         [0, 0, 0],
         [-1, -1, -1]],

        [[1, 1, 1],
         [0, 0, 0],
         [-1, -1, -1]],

        [[1, 1, 1],
         [0, 0, 0],
         [-1, -1, -1]]
    ]], dtype=torch.float32, device='cuda')
    '''
    kernel_x = torch.tensor([[[[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]],

                              [[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]],

                              [[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]]
                              ]], dtype=torch.float32, device='cuda')

    kernel_y = torch.tensor([[
        [[1, 1, 1],
         [0, 0, 0],
         [-1, -1, -1]],

        [[1, 1, 1],
         [0, 0, 0],
         [-1, -1, -1]],

        [[1, 1, 1],
         [0, 0, 0],
         [-1, -1, -1]]
    ]], dtype=torch.float32, device='cuda')

    # Set requires_grad to False to avoid gradients
    kernel_x = kernel_x.detach()
    kernel_y = kernel_y.detach()

    # Apply convolution
    edges_x = F.conv2d(image, kernel_x, stride=1, padding=1)
    edges_y = F.conv2d(image, kernel_y, stride=1, padding=1)

    # Calculate edge magnitude
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
    return edges


def solve(image,line):
    height, width = 640, 640
    # 解析标注信息并绘制边界框
    class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())

    # 转换为图像坐标
    x_center *= width
    y_center *= height
    box_width *= width
    box_height *= height

    # 计算左上角和右下角的坐标
    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)
    return shot_new(image, x1, x2, y1, y2)


def shot_new(image, x1, x2, y1, y2):
    crop = image[:,:, int(y1):int(y2), int(x1):int(x2)]
    return crop

def prewitt(paths, images):
    for i, path in enumerate(paths):
        directory, file_name = os.path.split(path)
        base, _ = os.path.splitext(file_name)
        new_dictionary = '/home/ycren/python/EVUP_part/trainAdir'
        new_file_path = os.path.join(new_dictionary, base + '.txt')
        with open(new_file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            edges = prewitt_operator(solve(images[i], line))


if __name__ == '__main__':
    path = '/home/test/123.jpg'
    dic, file = os.path.split(path)
    print(dic, file)
