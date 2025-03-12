import os


def clean_directory(directory):
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(directory):
        # 获取文件名和扩展名
        name, ext = os.path.splitext(filename)

        # 检查文件名是否以'_fake_B'结尾
        if not name.endswith('_fake'):
            # 删除不符合条件的文件
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

    # 再次遍历文件夹以去掉'_fake_B'后缀
    for filename in os.listdir(directory):
        name, ext = os.path.splitext(filename)

        # 如果文件名以'_fake_B'结尾，去掉后缀
        if name.endswith('_fake'):
            new_name = name[:-len('_fake')] + ext
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_name)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} to {new_file_path}")


import os
from PIL import Image


def convert_png_to_jpg(folder_path):
    # 遍历文件夹内的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            # 构造完整的文件路径
            png_file_path = os.path.join(folder_path, filename)
            jpg_file_path = os.path.join(folder_path, filename[:-4] + '.jpg')

            # 打开PNG文件并转换为JPG
            with Image.open(png_file_path) as img:
                # 将图像转换为RGB模式
                rgb_img = img.convert('RGB')
                # 保存为JPG文件
                rgb_img.save(jpg_file_path, 'JPEG')
                print(f'Converted: {png_file_path} to {jpg_file_path}')

import os

def delete_jpg_files(folder_path):
    # 遍历文件夹内的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件后缀名是否为.jpg
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)  # 删除文件
                print(f"已删除: {file_path}")
            except Exception as e:
                print(f"删除文件时出错: {file_path} - {e}")




if __name__ == '__main__':
    # 使用示例
    directory_path = '/home/ycren/shells/result_test_test/testparam/test_latest/images'  # 替换为你的文件夹路径
    clean_directory(directory_path)
    convert_png_to_jpg(directory_path)
    delete_jpg_files(directory_path)


