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
                img = cv2.resize(img, (640, 640))
                cv2.imwrite(os.path.join(output_path, 'test', filename), img)
                file.write(str(filename)+'\n')

                result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                avg_a = np.average(result[:, :, 1])
                avg_b = np.average(result[:, :, 2])
                result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
                result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
                result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

                cv2.imwrite(os.path.join(output_path,'WaterNet','wb',filename), result)


def ce(folder_path,output_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.xml') or filename.endswith('.txt'):
                continue
            # 读取彩色图像
            img = cv2.imread(os.path.join(folder_path, filename))
            img = cv2.resize(img, (640, 640))
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            cv2.imwrite(os.path.join(output_path, 'WaterNet', 'ce', filename), result)

def gc(folder_path,output_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.xml') or filename.endswith('.txt'):
                continue
            # 读取彩色图像
            img = cv2.imread(os.path.join(folder_path, filename))
            img = cv2.resize(img, (640, 640))
            gamma = 1.0
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            gamma_corrected = cv2.LUT(img, table)

            cv2.imwrite(os.path.join(output_path, 'WaterNet', 'gc', filename), gamma_corrected)
if __name__ == '__main__':
    output_path = '/home/ycren/python/UW/DATA/Test'

    os.mkdir(output_path+'/test')
    os.mkdir(output_path+'/WaterNet')
    os.mkdir(output_path+'/WaterNet/ce')
    os.mkdir(output_path+'/WaterNet/gc')
    os.mkdir(output_path+'/WaterNet/wb')

    folder_path = '/home/ycren/python/cycle_test/testA'
    wb(folder_path, output_path)
    ce(folder_path, output_path)
    gc(folder_path, output_path)