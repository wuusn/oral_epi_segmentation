from testWSI import downsize_slide
from testWSI import get_args
from utils.extract_anno import return_box
import time
import glob

def test_downsize_slide():
    start = time.time()
    src_path='/mnt/md0/_datasets/OralCavity/WSI/SFVA/SP07-1191 F4.tif' 
    save_path= \
        '/mnt/md0/_datasets/OralCavity/WSI/sfva/small_src/SP07-1191 F4.png'
    scale = 64
    downsize_slide(src_path, save_path, scale)
    end = time.time()

    print('test downsize_slide ok, time:', (end-start)/60)

def test_all_box():
    start = time.time()
    args  = get_args(True)
    src_dir = '/mnt/md0/_datasets/OralCavity/WSI/sfva/small_src' 
    mask_dir = '/mnt/md0/_datasets/OralCavity/WSI/SFVA/Masks' 
    src_paths = glob.glob(f'{src_dir}/*.png')
    for src_path in src_paths:
        name = src_path.split('/')[-1].replace('.png', '')
        print(name)
        mask_path = f'{mask_dir}/{name}_anno.png'
        return_box(src_path, mask_path, args)
    end = time.time()
    print((end-start)/60)


if __name__ == '__main__':
    #test_downsize_slide()
    test_all_box()

