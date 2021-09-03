import os
import glob

def compareDirFileSize(a , b):
    dirnames = glob.glob(f'{a}/*')
    a_size = {}
    for dirname in dirnames:
        files = glob.glob(f'{dirname}/*')
        size = len(files)
        name = dirname.split('/')[-1]
        a_size[name] = size
    a_size_sorted = dict(sorted(a_size.items(), key=lambda item: item[1]))
    for name,sizeA in a_size_sorted.items():
        files = glob.glob(f'{b}/{name}/*')
        sizeB = len(files)
        print(name, sizeA, sizeB)

if __name__ == '__main__':
    dirA = '/mnt/D/Oral/wsi_patch/SFVA'
    dirB = '/mnt/D/Oral/wsi_patch_edge/SFVA'
    compareDirFileSize(dirA, dirB)
