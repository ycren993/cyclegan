import numpy as np
from scipy.ndimage import convolve
from skimage.transform import resize
from matplotlib.image import imread
import os


# 定义 UICM 函数
def UICM(img):
    # 将图像的 RGB 通道分离并转换为浮点类型
    R = np.double(img[:, :, 0])
    G = np.double(img[:, :, 1])
    B = np.double(img[:, :, 2])

    # 计算 RG 和 YB 通道
    RG = R - G
    YB = (R + G) / 2 - B

    # 获取图像的像素总数
    K = RG.size

    # 处理 RG 通道
    RG1 = RG.flatten()  # 将二维数组展平为一维
    RG1 = np.sort(RG1)  # 对数组进行排序
    alphaL = 0.1
    alphaR = 0.1
    # 截取中间部分的值
    RG1 = RG1[int(alphaL * K): int(K * (1 - alphaR))]
    N = K * (1 - alphaL - alphaR)
    meanRG = np.sum(RG1) / N  # 计算均值
    deltaRG = np.sqrt(np.sum((RG1 - meanRG) ** 2) / N)  # 计算标准差

    # 处理 YB 通道
    YB1 = YB.flatten()  # 将二维数组展平为一维
    YB1 = np.sort(YB1)  # 对数组进行排序
    alphaL = 0.1
    alphaR = 0.1
    # 截取中间部分的值
    YB1 = YB1[int(alphaL * K): int(K * (1 - alphaR))]
    N = K * (1 - alphaL - alphaR)
    meanYB = np.sum(YB1) / N  # 计算均值
    deltaYB = np.sqrt(np.sum((YB1 - meanYB) ** 2) / N)  # 计算标准差

    # 计算 UICM
    uicm = -0.0268 * np.sqrt(meanRG ** 2 + meanYB ** 2) + 0.1586 * np.sqrt(deltaRG ** 2 + deltaYB ** 2)

    return uicm


# 定义 UISM 函数
def UISM(img):
    # 将图像的 RGB 通道分离并转换为浮点类型
    Ir = np.double(img[:, :, 0])
    Ig = np.double(img[:, :, 1])
    Ib = np.double(img[:, :, 2])

    # 定义 Sobel 梯度模板
    hx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    hy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # 计算每个通道的 Sobel 梯度
    SobelR = np.abs(convolve(Ir, hx, mode='reflect') + convolve(Ir, hy, mode='reflect'))
    SobelG = np.abs(convolve(Ig, hx, mode='reflect') + convolve(Ig, hy, mode='reflect'))
    SobelB = np.abs(convolve(Ib, hx, mode='reflect') + convolve(Ib, hy, mode='reflect'))

    patchsz = 5
    m, n = Ir.shape

    # 调整图像大小以匹配 patchsz
    if m % patchsz != 0 or n % patchsz != 0:
        new_m = m - (m % patchsz) + patchsz
        new_n = n - (n % patchsz) + patchsz
        SobelR = resize(SobelR, (new_m, new_n))
        SobelG = resize(SobelG, (new_m, new_n))
        SobelB = resize(SobelB, (new_m, new_n))

    m, n = SobelR.shape
    k1 = m // patchsz
    k2 = n // patchsz

    # 计算每个通道的 EME 值
    EMER = 0
    for i in range(0, m, patchsz):
        for j in range(0, n, patchsz):
            im = SobelR[i:i + patchsz, j:j + patchsz]
            if np.max(im) != 0 and np.min(im) != 0:
                EMER += np.log(np.max(im) / np.min(im))
    EMER = 2 / (k1 * k2) * np.abs(EMER)

    EMEG = 0
    for i in range(0, m, patchsz):
        for j in range(0, n, patchsz):
            im = SobelG[i:i + patchsz, j:j + patchsz]
            if np.max(im) != 0 and np.min(im) != 0:
                EMEG += np.log(np.max(im) / np.min(im))
    EMEG = 2 / (k1 * k2) * np.abs(EMEG)

    EMEB = 0
    for i in range(0, m, patchsz):
        for j in range(0, n, patchsz):
            im = SobelB[i:i + patchsz, j:j + patchsz]
            if np.max(im) != 0 and np.min(im) != 0:
                EMEB += np.log(np.max(im) / np.min(im))
    EMEB = 2 / (k1 * k2) * np.abs(EMEB)

    # 计算 UISM
    lambdaR = 0.299
    lambdaG = 0.587
    lambdaB = 0.114
    uism = lambdaR * EMER + lambdaG * EMEG + lambdaB * EMEB

    return uism


# 定义 UIConM 函数
def UIConM(img):
    # 将图像的 RGB 通道分离并转换为浮点类型
    R = np.double(img[:, :, 0])
    G = np.double(img[:, :, 1])
    B = np.double(img[:, :, 2])

    patchsz = 5
    m, n = R.shape

    # 调整图像大小以匹配 patchsz
    if m % patchsz != 0 or n % patchsz != 0:
        new_m = m - (m % patchsz) + patchsz
        new_n = n - (n % patchsz) + patchsz
        R = resize(R, (new_m, new_n))
        G = resize(G, (new_m, new_n))
        B = resize(B, (new_m, new_n))

    m, n = R.shape
    k1 = m // patchsz
    k2 = n // patchsz

    # 计算每个通道的 AMEE 值
    AMEER = 0
    for i in range(0, m, patchsz):
        for j in range(0, n, patchsz):
            im = R[i:i + patchsz, j:j + patchsz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEER += np.log((Max - Min) / (Max + Min)) * ((Max - Min) / (Max + Min))
    AMEER = 1 / (k1 * k2) * np.abs(AMEER)

    AMEEG = 0
    for i in range(0, m, patchsz):
        for j in range(0, n, patchsz):
            im = G[i:i + patchsz, j:j + patchsz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEEG += np.log((Max - Min) / (Max + Min)) * ((Max - Min) / (Max + Min))
    AMEEG = 1 / (k1 * k2) * np.abs(AMEEG)

    AMEEB = 0
    for i in range(0, m, patchsz):
        for j in range(0, n, patchsz):
            im = B[i:i + patchsz, j:j + patchsz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEEB += np.log((Max - Min) / (Max + Min)) * ((Max - Min) / (Max + Min))
    AMEEB = 1 / (k1 * k2) * np.abs(AMEEB)

    # 计算 UIConM
    uiconm = AMEER + AMEEG + AMEEB

    return uiconm


# 定义 UIQM 函数
def UIQM(image, c1=0.0282, c2=0.2953, c3=3.5753):
    uicm = UICM(image)
    uism = UISM(image)
    uiconm = UIConM(image)

    uiqm = c1 * uicm + c2 * uism + c3 * uiconm

    return uiqm


# 检测函数
def evaluate_image_quality(image_folder):
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    total_quality = 0

    for i in range(len(image_files)):
        # 确保文件是图像，而不是其他类型的文件（如 Thumbs.db）
        if not os.path.isdir(image_files[i]):
            # 读取当前图像文件
            img = imread(os.path.join(image_folder, image_files[i]))

            # 计算图像质量
            quality = UIQM(img)  # 假设 UIQM 是有效的图像质量评价函数

            # 累加质量评分
            total_quality += quality

            # 显示当前图像的质量和文件名
            print(f'Image {image_files[i]} quality score: {quality}')

    # 计算平均质量评分
    avg_quality = total_quality / len(image_files)

    # 显示平均质量评分信息
    print(f'所有图像的质量评价均值: {avg_quality}')


# 示例用法
if __name__ == "__main__":
    image_folder = 'djw37/ww'  # 替换为你的图像文件夹路径
    evaluate_image_quality(image_folder)
