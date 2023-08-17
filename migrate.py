import shutil
import os

src_folders = ['PaddleDetection/ppdet', 'PaddleCls/ppcls', 'PaddleSeg/paddleseg']
dst_folder = 'traintoolx'

# 将源文件夹转移到目标文件夹
for folder in src_folders:
    dir_name = folder.split('/')[1]
    dst_path = os.path.join(dst_folder, dir_name)

    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)

    shutil.copytree(folder, dst_folder)
