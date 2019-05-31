
import os
import sys
import shutil


# Returns the directory the current script (or interpreter) is running in
def get_script_directory():
    path = os.path.realpath(sys.argv[0])
    if os.path.isdir(path):
        return path
    else:
        return os.path.dirname(path)

# delete list of directories
def delete_create_dir(dir_list):
    for dir in dir_list: 
        if (os.path.exists(dir)):
            shutil.rmtree(dir)
            os.makedirs(dir)


# global variables
__DSET__ = 'dataset'
__TRAIN__ = 'train'
__VALID__ = 'valid'
__TEST__ = 'test'

__AUG__ = 'aug_img'
__KMOD__ = 'keras_model'
__TBLOG__ = 'tb_logs'


# image parameters
IMG_HEIGHT = 200
IMG_WIDTH = 250
IMG_CHAN = 3 # valid values - 1 or 3


