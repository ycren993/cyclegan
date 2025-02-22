import numpy as np
import cv2
import os
import torch

def np_list_int(tb):
    tb_2 = tb.tolist()  # 将np转换为列表
    return tb_2
# 读取图像
image_path = '/home/ycren/python/EVUP_part/testA/005244_jpg.rf.61ac949e7ebf7ec03ec7b67de10d362d.jpg'  # 图像路径
image = cv2.imread(image_path)
height, width, _ = image.shape

# 读取YOLO格式的标注文件
annotation_path = '/home/ycren/python/EVUP_part/testA/005244_jpg.rf.61ac949e7ebf7ec03ec7b67de10d362d.txt'  # yolo标注txt文件路径
with open(annotation_path, 'r') as file:
    lines = file.readlines()
def solve():
    # 解析标注信息并绘制边界框
    for i,line in enumerate(lines):
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
        shot_new(image_path,x1,x2,y1,y2,i)


def shot_new(img_path, x1,x2,y1,y2, i):
    print("加载图像： " + img_path)
    img = cv2.imread(img_path)
    crop = img[int(y1):int(y2), int(x1):int(x2)]

    cv2.imwrite("/home/ycren/python/testpic/" + str(i) + ".jpg", crop)  # 输出


if __name__ == '__main__':

    """

    文件夹下批量切割例子
    H1 为你要处理的文件夹

    """
    # for root, dirs, files in os.walk("H1"):
    #     print("图片列表：")
    #     print(files)
    #
    # left_up_1 = np.array([1323, 1810])  # 左上角坐标
    # left_down_1 = np.array([1323, 2190])  # 左下角坐标
    # right_up_1 = np.array([1943, 1810])  # 右上角坐标
    #
    # for num, val in enumerate(files):
    #     shot_new(img_path="H1/" + val, left_up=left_up_1, left_down=left_down_1, right_up=right_up_1, i=num)
    solve()

    
    # x = torch.randn(1, 3, 224, 224)

    # print(x[1])

