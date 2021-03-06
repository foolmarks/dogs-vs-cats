# Import modules
import keras
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator


import numpy as np
import pandas as pd
import os
import shutil
import argparse


import config

from customCNN import customCNN


# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


##############################################
# command line arguments
##############################################

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument('-b', '--batchsize',
                type=int,
                default=32,
	            help='training batchsize' )

ap.add_argument('-e', '--epochs',
                type=int,
                default=100,
	            help='number of training epochs')

ap.add_argument('-lr', '--learnrate',
                type=float,
                default=0.0001,
	            help='optimizer learning rate')

args = ap.parse_args()


print('\n----------------------------------------------')
print(' Command line options:')

print ('--batchsize: ', args.batchsize)
print ('--learnrate: ', args.learnrate)
print ('--epochs   : ', args.epochs)

print('----------------------------------------------')





##############################################
# Set up directories
##############################################

SCRIPT_DIR = config.get_script_directory()
print('This script is located in: ', SCRIPT_DIR)

TRAIN_DIR = os.path.join(SCRIPT_DIR, config._DSET_, config._TRAIN_)
VALID_DIR = os.path.join(SCRIPT_DIR, config._DSET_, config._VALID_)
TEST_DIR = os.path.join(SCRIPT_DIR, config._DSET_, config._TEST_)

# Augmented images folder
AUG_IMG_DIR = os.path.join(SCRIPT_DIR, config._AUG_)

# Keras model folder
KERAS_MODEL_DIR = os.path.join(SCRIPT_DIR, config._KMOD_)

# TensorBoard folder
TB_LOG_DIR = os.path.join(SCRIPT_DIR, config._TBLOG_)

# remove previous results
dir_list = [KERAS_MODEL_DIR, TB_LOG_DIR, AUG_IMG_DIR]
config.delete_create_dir(dir_list)

if (os.path.exists('results.csv')):
    os.remove('results.csv')


##############################################
# Training parameters
##############################################
# very unlikely to reach 100 epochs due to Early Stopping callback
EPOCHS = args.epochs

# batchsizes for training & validation
# batchsize for prediction is 1
BATCHSIZE = args.batchsize


# optimizer learning rate & decay rate
LEARN_RATE = args.learnrate
DECAY_RATE = LEARN_RATE/10.0

# assume we have either 3 channels or 1 channel
COLOR_MODE = 'rgb' if config.IMG_CHAN == 3 else 'grayscale'

##############################################
# CNN
##############################################
model = customCNN(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHAN))

print('\n----------------------------------------------')
print(' Model Summary')
print('----------------------------------------------')
# print a summary of the model
print(model.summary())
print("Model Inputs: {ips}".format(ips=(model.inputs)))
print("Model Outputs: {ops}".format(ops=(model.outputs)))


##############################################
# Input image pipeline for training, 
# and validation, prediction
##############################################

# data augmentation for training & validation
#   - pixel data is rescaled from 0:225 to 0:1.0
#   - random rotation of 5° max
#   - random horiz flip (images are flipped along vertical axis)
#   - random linear shift up and down by (200*01) pixels
datagen_tv = ImageDataGenerator(rescale=1/255,
                                rotation_range=5,
                                horizontal_flip=True,
                                height_shift_range=0.1,
                                width_shift_range=0.1
                                )

# data generation for prediction - only rescaling
datagen_p = ImageDataGenerator(rescale=1/255)

# train generator takes images from the specified directory, applies
# a resize (if required) with bilinear interpolation.
train_generator = datagen_tv.flow_from_directory(TRAIN_DIR,
                                                 target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
                                                 interpolation='bilinear',
                                                 batch_size=BATCHSIZE,
                                                 class_mode='binary',
                                                 color_mode=COLOR_MODE,
                                                 shuffle=True,
                                                 seed=42
                                                 )
'''
uncomment save_to_dir=AUG_IMG_DIR' to save the augmented images
note that this will take up considerable disk space
'''
validation_generator = datagen_tv.flow_from_directory(VALID_DIR,
                                                      target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
                                                      interpolation='bilinear',
                                                      batch_size=BATCHSIZE,
                                                      class_mode='binary',
                                                      color_mode=COLOR_MODE,
                                                      shuffle=True,
                                                    # save_to_dir=AUG_IMG_DIR
                                                      )


prediction_generator = datagen_p.flow_from_directory(TEST_DIR,
                                                     target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
                                                     interpolation='bilinear',
                                                     batch_size=1,
                                                     class_mode='binary',
                                                     color_mode=COLOR_MODE,
                                                     shuffle=False)


##############################################
# Compile model
##############################################
# Adam optimizer to change weights & biases
# Loss function is binary crossentropy
model.compile(optimizer=Adam(lr=LEARN_RATE, decay=0.0),
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])



##############################################
# Set up callbacks
##############################################
# create Tensorboard callback
tb_call = TensorBoard(log_dir=TB_LOG_DIR,
                      batch_size=BATCHSIZE)

'''
Early stop callback to halt training if validation accuracy 
stops improving for 5 epochs.
The weights from the epoch that gives best results are restored 
once training stops.
'''
earlystop_call = EarlyStopping(monitor='val_binary_accuracy',
                               mode='max',
                               min_delta=0.0001,
                               patience=5,
                               restore_best_weights=True)


callbacks_list = [tb_call, earlystop_call]

print('\n----------------------------------------------')
print(' Training model with training set..')
print('----------------------------------------------')

# calculate number of steps in one training epoch
train_steps = train_generator.n//train_generator.batch_size

# calculate number of steps in one validation epoch
val_steps = validation_generator.n//validation_generator.batch_size

# run training
train_history=model.fit_generator(generator=train_generator,
                                  epochs=EPOCHS,
                                  steps_per_epoch=train_steps,
                                  validation_data=validation_generator,
                                  validation_steps=val_steps,
                                  callbacks=callbacks_list,
                                  shuffle=True)


print("\nTensorBoard can be opened with the command: tensorboard --logdir={dir} --host localhost --port 6006".format(dir=TB_LOG_DIR))

print('\n----------------------------------------------')
print(' Evaluate model accuracy with validation set..')
print('----------------------------------------------')
scores = model.evaluate_generator(generator=validation_generator,
                                  max_queue_size=10,
                                  steps=val_steps,
                                  verbose=1)

print ('Evaluation Loss    : ', scores[0])
print ('Evaluation Accuracy: ', scores[1])


print('\n----------------------------------------------')
print(' Test predictions accuracy with test dataset..')
print('----------------------------------------------')

# reset the generator before using it for predictions
prediction_generator.reset()

# get a list of image filenames used in prediction
filenames = prediction_generator.filenames

# calculate number of steps for prediction
predict_steps = prediction_generator.n

# predict generator returns a list of all predictions
pred = model.predict_generator(generator=prediction_generator,
                               steps=predict_steps,
                               verbose=1)

# the .class_indices attribute returns a dictionary with keys = classes
labels = (prediction_generator.class_indices)


# make a new dictionary with keys & values swapped 
labels = dict((v,k) for k,v in labels.items())


# use the 'labels dictionary to create a list of predictions as strings
# predictions is a list of sigmoid outputs
# if sigmoid output < 0.5, CNN predicted class 0
# if sigmoid output > 0.5, CNN predicted class 1
predictions = list()
for i in range(len(pred)):
    if pred[i] > 0.5:
        predictions.append(labels[1])
    else:   
        predictions.append(labels[0])

# iterate over the list of predictions and compare to ground truth labels
# ground truth labels are derived from prediction filenames.
correct_predictions = 0
wrong_predictions = 0

for i in range (len(predictions)):

    # ground truth is first part of filename (i.e. the class folder)
    # will need to be modified to '\' for windows
    ground_truth, _ = filenames[i].split('/',1)

    # compare prediction to ground truth
    if predictions[i] == ground_truth:
        correct_predictions += 1
    else:
        wrong_predictions += 1

# calculate accuracy
acc = (correct_predictions/len(predictions)) * 100

print('Correct Predictions: ',correct_predictions)
print('Wrong Predictions  : ',wrong_predictions)
print('Prediction Accuracy: ',acc,'%')


# write filenames and associated predictions to .csv file
results = pd.DataFrame({"Filename":filenames,
                        "Predictions":predictions})
results.to_csv('results.csv',index=False)

print('\nPredictions and true labels saved to results.csv')


print('\n----------------------------------------------')
print(' Saving trained model..')
print('----------------------------------------------')

# save just the weights (no architecture) to an HDF5 format file
model.save_weights(os.path.join(KERAS_MODEL_DIR,'k_model_weights.h5'))

# save just the architecture (no weights) to a JSON file
with open(os.path.join(KERAS_MODEL_DIR,'k_model_architecture.json'), 'w') as f:
    f.write(model.to_json())

print('\nTrained model saved to {dir}'.format(dir=KERAS_MODEL_DIR))


print('\n----------------------------------------------')
print(' FINISHED!')
print('----------------------------------------------')








