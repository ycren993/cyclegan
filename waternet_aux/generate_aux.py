import os

import cv2
import numpy as np
def wb(folder_path,output_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        with open(os.path.join(output_path, 'test.txt'), 'w') as file:
            for filename in os.listdir(folder_path):
                if filename.endswith('.xml') or filename.endswith('.txt'):
                    continue
                # 读取彩色图像
                img = cv2.imread(os.path.join(folder_path,filename))

                file.write(str(filename)+'\n')
                # 计算平均亮度
                average_blue = np.mean(img[:, :, 1])
                average_green = np.mean(img[:, :, 2])
                average_red = np.mean(img[:, :, 0])

                # 计算权重因子
                w_blue = 1.0 / average_blue * average_green
                w_green = 1.0 / average_green * average_red
                w_red = 1.0 / average_red * average_blue

                # 应用权重因子进行白平衡校正
                img_corrected = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式以便处理
                img_corrected[:, :, 0] = w_blue * img[:, :, 0] + (1 - w_blue) * img[:, :, 2]  # 蓝色通道校正
                img_corrected[:, :, 1] = w_green * img[:, :, 1]  # 绿色通道保持不变
                img_corrected[:, :, 2] = w_red * img[:, :, 2] + (1 - w_red) * img[:, :, 0]  # 红色通道校正
                img_corrected = cv2.cvtColor(img_corrected, cv2.COLOR_RGB2BGR)  # 转回BGR格式以便显示和保存
                cv2.imwrite(os.path.join(output_path,'WaterNet','wb',filename), img_corrected)
                cv2.imwrite(os.path.join(output_path,'test',filename), img)

def ce(folder_path,output_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.xml') or filename.endswith('.txt'):
                continue
            # 读取彩色图像
            img = cv2.imread(os.path.join(folder_path, filename))

            # 将彩色图像分解为单独的颜色通道
            b, g, r = cv2.split(img)

            # 对每个颜色通道进行直方图均衡化
            b_eq = cv2.equalizeHist(b)
            g_eq = cv2.equalizeHist(g)
            r_eq = cv2.equalizeHist(r)

            # 合并均衡化后的颜色通道
            result = cv2.merge((b_eq, g_eq, r_eq))
            cv2.imwrite(os.path.join(output_path, 'WaterNet', 'ce', filename), result)
def gc(folder_path,output_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.xml') or filename.endswith('.txt'):
                continue
            # 读取彩色图像
            img = cv2.imread(os.path.join(folder_path, filename))
            gamma = 0.5  # 可以根据需要调整gamma值，gamma<1会使图像变亮，gamma>1会使图像变暗
            lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
            gamma_corrected = cv2.LUT(img, lookup_table)  # 应用伽马校正后的查找表进行校正处理后的图像显示和保存。
            cv2.imwrite(os.path.join(output_path, 'WaterNet', 'gc', filename), gamma_corrected)
if __name__ == '__main__':
    os.mkdir('/home/ycren/python/UW/DATA/Test/test')
    os.mkdir('/home/ycren/python/UW/DATA/Test/WaterNet')
    os.mkdir('/home/ycren/python/UW/DATA/Test/WaterNet/ce')
    os.mkdir('/home/ycren/python/UW/DATA/Test/WaterNet/gc')
    os.mkdir('/home/ycren/python/UW/DATA/Test/WaterNet/wb')

    folder_path = '/home/ycren/python/trash_ICRA19/dataset/test'
    output_path = '/home/ycren/python/UW/DATA/Test'
    wb(folder_path, output_path)
    ce(folder_path, output_path)
    gc(folder_path, output_path)