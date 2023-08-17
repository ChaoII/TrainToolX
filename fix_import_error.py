import os
import shutil


def rename_imports(file_path):
    # 读取文件内容
    with open(file_path, 'r', encoding="utf8") as f:
        content = f.readlines()

    # 替换import语句
    new_content = ""
    for line in content:
        if line.startswith("import paddleseg"):
            new_line = line.replace("import paddleseg", "import traintoolx.paddleseg")
            new_content += new_line
        elif line.startswith("from paddleseg"):
            new_line = line.replace("from paddleseg", "from traintoolx.paddleseg")
            new_content += new_line
        elif line.startswith("import ppcls"):
            new_line = line.replace("import ppcls", "import traintoolx.ppcls")
            new_content += new_line
        elif line.startswith("from ppcls"):
            new_line = line.replace("from ppcls", "from traintoolx.ppcls")
            new_content += new_line

        elif line.startswith("import ppdet"):
            new_line = line.replace("import ppdet", "import traintoolx.ppdet")
            new_content += new_line

        elif line.startswith("from ppdet"):
            new_line = line.replace("from ppdet", "from traintoolx.ppdet")
            new_content += new_line

        else:
            new_content += line

    # 将修改后的内容写回文件
    with open(file_path, 'w', encoding="utf8") as f:
        f.writelines(new_content)


def getfiles(dst_dir):
    for ems in os.listdir(dst_dir):
        file_path = os.path.join(dst_dir, ems)
        print(file_path)
        if os.path.isdir(file_path):
            getfiles(file_path)
        else:
            if ems.endswith('.py'):
                # 重命名导入语句
                rename_imports(file_path)
    print("替换完成")


if __name__ == '__main__':
    dst_dir_s = ["traintoolx/paddleseg", "traintoolx/ppdet", "traintoolx/ppcls"]
    for dst_dir_ in dst_dir_s:
        getfiles(dst_dir_)
