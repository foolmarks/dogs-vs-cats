# Dogs-vs-Cats with Keras
Keras example of Kaggle's dogs-vs-cats


For more details, see my website: http://www.markharvey.info/ml/dogs_cats/dogs_cats.html

## Instructions

1. Clone or download the repository.

2. Download dataset from https://www.kaggle.com/c/dogs-vs-cats/data.  The downloaded zip file should be placed in the same folder as the python (.py) scripts.

3. Run create_datasets.py - only needs to be run once.

4. Run train.py - executes training, validation and prediction accuracy testing of the trained model. It saves the trained model in ./keras_model.

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
