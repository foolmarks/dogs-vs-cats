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
import shutil
import zipfile
import random

import config

print('###############################################')
print(' DATASET PREPARATION STARTED..')
print('###############################################')

SCRIPT_DIR = config.get_script_directory()
print('\nThis script is located in: ', SCRIPT_DIR)

###############################################
# make the required folders
###############################################
# dataset top level
DATASET_DIR = os.path.join(SCRIPT_DIR, config.__DSET__)

# train, validation and test folders
TRAIN_DIR = os.path.join(DATASET_DIR, config.__TRAIN__)
VALID_DIR = os.path.join(DATASET_DIR, config.__VALID__)
TEST_DIR = os.path.join(DATASET_DIR, config.__TEST__)

# class folders
TRAIN_CAT_DIR = os.path.join(TRAIN_DIR, 'cat')
TRAIN_DOG_DIR = os.path.join(TRAIN_DIR, 'dog')
VALID_CAT_DIR = os.path.join(VALID_DIR, 'cat')
VALID_DOG_DIR = os.path.join(VALID_DIR, 'dog')
TEST_CAT_DIR = os.path.join(TEST_DIR, 'cat')
TEST_DOG_DIR = os.path.join(TEST_DIR, 'dog')


# remove any previous data
dir_list = [DATASET_DIR]
config.delete_create_dir(dir_list)


# unzip the dogs-vs-cats archive that was downloaded from Kaggle
zip_ref = zipfile.ZipFile('./dogs-vs-cats.zip', 'r')
zip_ref.extractall(DATASET_DIR)
zip_ref.close()

# unzip train archive (inside the dogs-vs-cats archive)
zip_ref = zipfile.ZipFile(os.path.join(DATASET_DIR, 'train.zip'), 'r')
zip_ref.extractall(DATASET_DIR)
zip_ref.close()

print('\nUnzipped dataset..\n')

# make all necessary folders
dir_list = [VALID_DIR,TEST_DIR,TRAIN_CAT_DIR,TRAIN_DOG_DIR, \
            VALID_CAT_DIR,VALID_DOG_DIR,TEST_CAT_DIR,TEST_DOG_DIR]
 
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

# shuffle the list of images
random.shuffle(imageList)


# make lists of images according to their class
catImages=list()
dogImages=list()

for img in imageList:
    filename = os.path.basename(img)
    class_name,_ = filename.split('.',1)
    if class_name == 'cat':
        catImages.append(img)
    else:
        dogImages.append(img)


# define train, valid, test split as 70:20:10
testImages = int(len(dogImages) * 0.1)
validImages = int(len(dogImages) * 0.3)


# move the images to their class folders inside train, valid, test
for i in range(0,testImages):
    filename_d = os.path.basename(dogImages[i])
    filename_c = os.path.basename(catImages[i])
    os.rename(dogImages[i], os.path.join(TEST_DOG_DIR, filename_d))
    os.rename(catImages[i], os.path.join(TEST_CAT_DIR, filename_c))

for i in range(testImages,validImages):
    filename_d = os.path.basename(dogImages[i])
    filename_c = os.path.basename(catImages[i])
    os.rename(dogImages[i], os.path.join(VALID_DOG_DIR, filename_d))
    os.rename(catImages[i], os.path.join(VALID_CAT_DIR, filename_c))

for i in range(validImages,len(dogImages)):
    filename_d = os.path.basename(dogImages[i])
    filename_c = os.path.basename(catImages[i])
    os.rename(dogImages[i], os.path.join(TRAIN_DOG_DIR, filename_d))
    os.rename(catImages[i], os.path.join(TRAIN_CAT_DIR, filename_c))


# run a check on number of files in each class folder
dir_list = [TEST_DOG_DIR,TEST_CAT_DIR,VALID_DOG_DIR,VALID_CAT_DIR,TRAIN_DOG_DIR,TRAIN_CAT_DIR]
for dir in dir_list: 
    file_count = sum([len(files) for root,dir,files in os.walk(dir)])
    print('Number of Files in', dir,': ',file_count)

print('\n###############################################')
print(' DATASET PREPARATION FINISHED')
print('###############################################')

