# Dogs-vs-Cats with Keras
Keras example of Kaggle's dogs-vs-cats


For more details, see my website: http://www.markharvey.info/ml/dogs_cats/dogs_cats.html

## Instructions

1. Clone or download this repository.

2. Download the dataset from https://www.kaggle.com/c/dogs-vs-cats/data.  The downloaded zip file should be placed in the same folder as the python (.py) scripts.

3. If you are working from within a python virtual environment, activate it.

4. Run create_datasets.py - must be run at least once. Does not need to be run again if no change to the dataset.

5. Run train.py - executes training, validation and prediction accuracy testing of the trained model. It saves the trained model in ./keras_model.


## Dependencies

The python code has been written for, and only tested with, python 3.6.

All python scripts were run from within a virtual environment created with Anaconda3. The environment YAML file is included in this repository so that it can be recreated.



## Command line arguments

Note that train.py has 3 optional command line arguments:


| Argument               | Type     | Default value| Description                             |  
| ---------------------- | ---------|:------------:| ----------------------------------------|  
| `--batchsize` or `-b`  | integer  |   32         | The training data batchsize             |  
| `--learnrate` or `-lr` | float    |   0.0001     | The learning rate used by the optimizer |  
| `--epochs`    or `-e`  | integer  |   100        | The number of training epochs           |  


Example command lines:

Train using the default values for batchsize, epochs and learning rate:<br>
`python train.py`


Train for 20 epochs, use default values for batchsize and learning rate:<br>
`python train.py --epochs 20`


Train using learning rate of 0.001, use default values for batchsize and epochs:<br>
`python train.py -lr 0.001` 


Train using batchsize of 64, use default values for learning rate and epochs:<br>
`python train.py -b 64`


Train for 10 epochs using batchsize of 50, use default value for learning rate:<br>
`python train.py --batchsize 50 --epochs 10` 


## Jupyter NoteBooks

There are two Jupyter NoteBooks included in the repository. Their main purpose is to provide some step-by-step explanations of how the design works, they do not substitute the python scripts.


