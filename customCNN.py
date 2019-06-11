'''
Simple custom CNN
There are no pooling layers, instead convolution layers with 
2x2 kernels and stride length = 2 are used for data reduction.
Dense layers are also replaced with convolution layers.
Sigmoid activation in the final layer.
'''

from keras.models import Model
from keras import regularizers
from keras.layers import Input, Flatten, Conv2D
from keras.layers import BatchNormalization, Activation


# wrapper function for the 2D convolution layer
def conv_layer(input, filters, kernel_size, strides=1):

      return Conv2D(filters=filters, kernel_size=kernel_size, kernel_initializer='he_uniform', \
                    kernel_regularizer=regularizers.l2(0.01), \
                    strides=strides, activation='relu', padding='same')(input)



def customCNN(input_shape=(200, 200, 3)):

      input_layer = Input(shape=input_shape)

      net = conv_layer(input_layer, 32, 3)
      net = conv_layer(net, 32, 2, 2)
      net = BatchNormalization()(net)

      net = conv_layer(net, 64, 3)
      net = conv_layer(net, 64, 2, 2)
      net = BatchNormalization()(net)

      net = conv_layer(net, 128, 3)
      net = conv_layer(net, 128, 2, 2)
      net = BatchNormalization()(net)

      net = conv_layer(net, 128, 5, 5)
      net = conv_layer(net, 1, 5, 7)
      net = BatchNormalization()(net)
      net = Flatten()(net)
      output_layer = Activation('sigmoid')(net)
     
      return Model(inputs=input_layer, outputs=output_layer)
