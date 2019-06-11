
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

# delete & recreate list of directories
def delete_create_dir(dir_list):
    for dir in dir_list: 
        if (os.path.exists(dir)):
            shutil.rmtree(dir)
        os.makedirs(dir)


# global variables
_DSET_ = 'dataset'
_TRAIN_ = 'train'
_VALID_ = 'valid'
_TEST_ = 'test'

_AUG_ = 'aug_img'
_KMOD_ = 'keras_model'
_TBLOG_ = 'tb_logs'


# image parameters
IMG_HEIGHT = 200
IMG_WIDTH = 250
IMG_CHAN = 3 # valid values - 1 or 3


