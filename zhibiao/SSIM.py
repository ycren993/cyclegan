import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def psnr_ssim_helper(original_path,reconstructed_path):
    # 读取原始图像和重建图像
    original_image = cv2.imread(original_path)
    reconstructed_image = cv2.imread(reconstructed_path)

    # 计算均方误差（MSE）
    mse = np.mean((original_image - reconstructed_image) ** 2)

    # 计算PSNR
    max_pixel_value = 255  # 对于8位图像，最大像素值为255
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)

    ssim_data = ssim(original_image, reconstructed_image,channel_axis=2)

    # print(f"PSNR: {psnr} dB")
    return psnr, ssim_data
def PSNR(original_fold, reconstructed_fold_waternet, reconstructed_fold_cyclegan):
    sum = 0
    ok_psnr = 0
    ok_ssim = 0
    sum_psnr_waternet = 0.0
    sum_ssim_waternet = 0.0
    sum_psnr_cyclegan = 0.0
    sum_ssim_cyclegan = 0.0
    for filename in os.listdir(original_fold):
        sum = sum + 1
        psnr_cyc, ssim_cyc = psnr_ssim_helper(os.path.join(original_fold, filename), os.path.join(reconstructed_fold_cyclegan, filename))
        psnr_water, ssim_water = psnr_ssim_helper(os.path.join(original_fold, filename), os.path.join(reconstructed_fold_waternet, filename) \
                                                  # .replace('.jpg','.png') \
                                                  )
        sum_psnr_cyclegan = sum_psnr_cyclegan + psnr_cyc
        sum_ssim_cyclegan = sum_ssim_cyclegan + ssim_cyc
        sum_psnr_waternet = sum_psnr_waternet+psnr_water
        sum_ssim_waternet = sum_ssim_waternet+ssim_water
        if psnr_cyc >= psnr_water:
            ok_psnr = ok_psnr + 1
        if abs(1-ssim_cyc) <= abs(1-ssim_water):
            ok_ssim = ok_ssim + 1
    print('sum_psnr_cyclegan = ',sum_psnr_cyclegan/sum)
    print('sum_ssim_cyclegan = ',sum_ssim_cyclegan/sum)
    print('sum_psnr_waternet = ',sum_psnr_waternet/sum)
    print('sum_ssim_waternet = ',sum_ssim_waternet/sum)
    return ok_psnr, ok_ssim, sum


if __name__ == '__main__':
    ok_psnr, ok_ssim, sum = PSNR('/home/ycren/python/URPC-Official/test/images', '/home/ycren/python/FUnIE-GAN-master/output_urpc_uiqm2.6394_input2.055', '/home/ycren/shells/result_test_test/testparam/test_latest/images')
    print('ok_psnr = ',ok_psnr)
    print('ok_ssim = ',ok_ssim)
    print('sum = ',sum)
