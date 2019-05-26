'''
TESTED WITH PYTHON 3.6

This script will do data wrangling on the dogs-vs-cats dataset.

The dataset must be downloaded from https://www.kaggle.com/c/dogs-vs-cats/data
 - this will require a Kaggle account.

The downloaded 'dogs-vs-cats.zip' archive should be placed in the same folder 
as this script, then this script should be run.

The script will create a folder structure compatible with Keras's 
ImageGenerator.flow_from_directory():

dataset
   |_dogs-vs-cats
       |_test
       |   |_cats
       |   |_dogs
       |_train
       |   |_cats
       |   |_dogs
       |_valid
           |_cats
           |_dogs

'''

import os
import sys
import shutil
import cv2

import zipfile

from random import seed
from random import random


# Returns the directory the current script (or interpreter) is running in
def get_script_directory():
    path = os.path.realpath(sys.argv[0])
    if os.path.isdir(path):
        return path
    else:
        return os.path.dirname(path)

SCRIPT_DIR = get_script_directory()
print('This script is located in: ', SCRIPT_DIR)

###############################################
# make the required folders
###############################################
# dataset top level
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')

# train, validation and test folders
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VALID_DIR = os.path.join(DATASET_DIR, 'valid')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

# class folders
TRAIN_CAT_DIR = os.path.join(TRAIN_DIR, 'cat')
TRAIN_DOG_DIR = os.path.join(TRAIN_DIR, 'dog')
VALID_CAT_DIR = os.path.join(VALID_DIR, 'cat')
VALID_DOG_DIR = os.path.join(VALID_DIR, 'dog')
TEST_CAT_DIR = os.path.join(TEST_DIR, 'cat')
TEST_DOG_DIR = os.path.join(TEST_DIR, 'dog')


# remove any previous data
dir_list = [DATASET_DIR]
for dir in dir_list: 
    if (os.path.exists(dir)):
        shutil.rmtree(dir)
    os.makedirs(dir)
    print("Directory" , dir ,  "created ")

# unzip the dogs-vs-cats archive that was downloaded from Kaggle
zip_ref = zipfile.ZipFile('./dogs-vs-cats.zip', 'r')
zip_ref.extractall('./dataset')
zip_ref.close()

# unzip train archive (inside the dogs-vs-cats archive)
zip_ref = zipfile.ZipFile('./dataset/train.zip', 'r')
zip_ref.extractall('./dataset')
zip_ref.close()

print('Unzipped dataset..')

# make all necessary folders
dir_list = [VALID_DIR, TEST_DIR,TRAIN_CAT_DIR,TRAIN_DOG_DIR, \
            VALID_CAT_DIR, VALID_DOG_DIR,TEST_CAT_DIR,TEST_DOG_DIR]
 
for dir in dir_list: 
    os.makedirs(dir)
    print("Directory " , dir ,  "created ")


# remove un-needed files
os.remove(os.path.join(DATASET_DIR, 'sampleSubmission.csv'))
os.remove(os.path.join(DATASET_DIR, 'test1.zip'))
os.remove(os.path.join(DATASET_DIR, 'train.zip'))


###############################################
# move the images to class folders
###############################################
# make a list of all files currently in the train folder
imageList = list()
for (root, name, files) in os.walk(TRAIN_DIR):
    imageList += [os.path.join(root, file) for file in files]


# seed random number generator
seed(1)
# define ratio of pictures to use for train, test, validation
# approx 4% of images will be sent to the test folder, 16% to the 
# validation folder, all others go to train folder
test_ratio = 0.04
valid_ratio = 0.2


# move the images to their class folders inside train, valid, test
for img in imageList:
    filename = os.path.basename(img)
    class_folder,_ = filename.split('.',1)

    # choose between train, test, validation based on random number
    if random() <= test_ratio:
        dst_dir = TEST_DIR
    elif (random() > test_ratio and random() <= (test_ratio + valid_ratio)):
        dst_dir = VALID_DIR
    else:
        dst_dir = TRAIN_DIR
       
    os.rename(img, os.path.join(dst_dir, class_folder, filename))

print ('FINISHED CREATING DATASET')

