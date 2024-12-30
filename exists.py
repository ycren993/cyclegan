import os


def check_files(jpg_folder, txt_folder):
    # 获取jpg文件夹中的所有jpg文件
    jpg_files = [f for f in os.listdir(jpg_folder) if f.endswith('.jpg')]

    # 遍历每个jpg文件，检查是否存在同名的txt文件
    for jpg_file in jpg_files:
        # 获取文件名（不带扩展名）
        file_name = os.path.splitext(jpg_file)[0]
        # 构造对应的txt文件名
        txt_file = file_name + '.txt'

        # 检查txt文件是否存在
        if txt_file in os.listdir(txt_folder):
            # print(f"Found matching TXT for {jpg_file}: {txt_file}")
            pass
        else:
            print(f"No matching TXT found for {jpg_file}")


# 示例用法
jpg_folder_path = '/home/ycren/python/EVUP_part/trainA/'  # 替换为你的jpg文件夹路径
txt_folder_path = '/home/ycren/python/EVUP_part/trainAdir/'  # 替换为你的txt文件夹路径

check_files(jpg_folder_path, txt_folder_path)