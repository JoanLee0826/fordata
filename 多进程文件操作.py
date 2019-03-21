import os
import zipfile
import re
from unrar import rarfile


def scan(filepath):

    for fpath in os.listdir(filepath):
        fpath = filepath + '/' + fpath
        if os.path.isdir(fpath):
            scan(fpath)
        else:
            file_handle(file=fpath)


def file_handle(file):
    fpath, fname = os.path.split(file)
    if fname.endswith('.rar') and re.match(r'\d+', fname):
        f = rarfile.RarFile(file)
        f.extractall(fpath)


if __name__ == '__main__':
    filepath = input("输入文件夹路径：")
    scan(filepath)
    print("文件操作完成")