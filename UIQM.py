import math
import cv2
from  skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import numpy as np
def calculate_ssim(image_path1, image_path2):
    """
    Calculate the SSIM between two images.

    Parameters:
    image_path1 (str): The path to the first image.
    image_path2 (str): The path to the second image.

    Returns:
    float: The SSIM score.
    """
    # Read the images using OpenCV
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Check if the images were successfully loaded
    if image1 is None or image2 is None:
        raise ValueError("Error: One or both images could not be loaded.")

    # Convert the images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    ssim_score, _ = ssim(gray_image1, gray_image2, full=True)

    return ssim_score
def getUCIQE(img):
    img_BGR = cv2.imread(img)
    img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
    img_LAB = np.array(img_LAB, dtype=np.float64)
    # 根据论文中给出的训练系数<An Underwater Color Image Quality Evaluation Metric>
    coe_Metric = [0.4680, 0.2745, 0.2576]

    img_lum = img_LAB[:, :, 0] / 255.0
    img_a = img_LAB[:, :, 1] / 255.0
    img_b = img_LAB[:, :, 2] / 255.0

    # 色度标准差
    chroma = np.sqrt(np.square(img_a) + np.square(img_b))
    sigma_c = np.std(chroma)

    # 亮度对比度
    img_lum = img_lum.flatten()
    sorted_index = np.argsort(img_lum)
    top_index = sorted_index[int(len(img_lum) * 0.99)]
    bottom_index = sorted_index[int(len(img_lum) * 0.01)]
    con_lum = img_lum[top_index] - img_lum[bottom_index]

    # 饱和度均值
    chroma = chroma.flatten()
    sat = np.divide(chroma, img_lum, out=np.zeros_like(chroma, dtype=np.float64), where=img_lum != 0)
    avg_sat = np.mean(sat)

    uciqe = sigma_c * coe_Metric[0] + con_lum * coe_Metric[1] + avg_sat * coe_Metric[2]
    return uciqe

import os
def read_files(folder_path):
    # Specify the path to the folder
    extra_path = '/home/ycren/python/pytorch-CycleGAN-and-pix2pix/results/resnet_9blocks_cyclegan/test_latest/images'
    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Loop through all files in the folder
        i = 0
        sum = 0
        j = 0
        for filename in os.listdir(folder_path):
            sum = sum + 1
            # Check if it is a file
            if os.path.isfile(os.path.join(folder_path, filename)):
                # Split the filename and remove the extension
                name, _ = os.path.splitext(filename)
                # print('concat: ',name.split('_')[-2] +'_'+name.split('_')[-1], 'name: ', name)
                # print(name)
                concat_name = name.split('_')[-2] +'_'+name.split('_')[-1]
                if concat_name == 'real_A':
                    print('------------------------------------')
                    print('txt路径: ', os.path.join(folder_path, name+'.txt'))
                    real_A_path = os.path.join(folder_path, filename)
                    print('real_A路径: ', real_A_path,'UCIQE: ', getUCIQE(real_A_path))
                    # if os.path.exists(real_A_path):
                    #     print('real_A路径合法')
                    extra_real = os.path.join(extra_path, filename)
                    extra_fake = os.path.join(extra_path, filename.replace('real_A', 'fake_B'))
                    fake_B_path = os.path.join(folder_path, filename.replace('real_A', 'fake_B'))
                    uciqe_null = getUCIQE(extra_fake)
                    uciqe_suanzi = getUCIQE(fake_B_path)
                    print('fake_B路径: ', fake_B_path,'UCIQE: ', uciqe_suanzi)
                    print('extra_B路径: ', extra_fake,'UCIQE: ', uciqe_null)
                    # print('ssim_算子: ', calculate_ssim(real_A_path, fake_B_path))

                    ssim_suanzi = calculate_ssim(real_A_path, fake_B_path)
                    ssim_null =calculate_ssim(extra_real,extra_fake)
                    print('ssim_算子: ', ssim_suanzi)
                    print('ssim_null: ',ssim_null)
                    if ssim_null <= ssim_suanzi:
                        i = i+1
                    if uciqe_null <= uciqe_suanzi:
                        j = j +1
                    # if os.path.exists(fake_B_path):
                    #     print('fake_B路径合法')
        print('i = ',i, ', sum = ', sum)
        print('j = ', j, ', sum = ', sum)
    else:
        print("The specified folder does not exist or is not a directory.")


if __name__ == '__main__':
    # img = cv2.imread(r'/home/ycren/python/pytorch-CycleGAN-and-pix2pix/results/test_debug_cut/test_latest/images/005220_jpg.rf.4af5a3ab3f432bb24d479f4669a7d2b4_real_A.png')
    # print(getUCIQE(r'/home/ycren/python/pytorch-CycleGAN-and-pix2pix/results/test_debug_idt/test_latest/images/005215_jpg.rf.98d25ad03e0302ae5afd1551d4888dd2_fake_B.png'))
    read_files(r'/home/ycren/python/pytorch-CycleGAN-and-pix2pix/results/P_SE_ResNet_blocks/test_latest/images')