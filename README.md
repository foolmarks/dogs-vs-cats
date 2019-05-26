# Dogs-vs-Cats with Keras
Keras example of Kaggle's dogs-vs-cats

## Instructions

1. Clone or download the repository.

2. Download dataset from https://www.kaggle.com/c/dogs-vs-cats/data.  The downloaded zip file should be placed in the same folder as the python (.py) scripts.

3. Run create_datasets.py - only needs to be run once.

4. Run train.py - executes training, validation and prediction accuracy testing of the trained model. It saves the trained model in ./keras_model.


## Dataset preparation
The dogs-vs-cats.zip archive will be unzipped by the create_datasets.py script. This script will also create all of the necessary folders to make the data compatible with the format required by the Keras .flow_from_directory() method.

The script uses a random number generator to choose which image files will be used for training, validation and prediction.

The images in the test folder are only used for prediction to guarantee that they are 'unseen' data.


## The Convolution Neural Network

The customCNN.py script uses the Keras Functional API to describe the simple CNN. The CNN is *fully-convolutional* - the dense or fully-connected layers have been replaced with convolutional layers that have their kernel sizes, number of filters and stride lengths set such that they create output shapes that mimic the output shapes of dense/FC layers.

There are no pooling layers - these have also been replaced with convolutional layers that have their kernel size and strides set to the same value which is > 1.

The output activation layer is a sigmoid function as we only have two classes - if the output of the sigmoid is > 0.5, the predicted class is 'dog', less that 0.5 is a prediction of 'cat'.

The CNN has deliberately been kept simple (it only has 8 convolutional layers) so the expected prediction accuracy will not be higher than approximately 90%.

To reduce overfitting, batch normalization layers have been used and also L2 kernel regularization.

## Training features

The train.py script executes the training, evaluation and prediction accuracy testing and uses some advanced features of Keras:

+ Images are read from disk using the flow_from_directory() method.
+ On-the-fly image augmentation is used:
  + Normalization of pixel values from 0:255 to 0:1
  + Images are resized to 200 x 200 using bilinear interpolation.
  + Random flipping along the vertical axis.
  + Random vertical and horizontal shifts.
  + Shuffling of the images between epochs.
+ Early stopping of training if the validation accuracy stops increasing for a certain number of epochs.
  + The CNN parameters from the epoch with the best validation accuracy are automatically restored after training stops.
+ Saving of the trained model as JSON and HDF5 files.
+ Prediction results saved to a .csv file using Pandas.
