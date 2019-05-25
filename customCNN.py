'''
Simple custom CNN with sigmoid activation in the
final layer.

There are no pooling layers, instead convolution layers with 
2x2 kernels and stride length = 2 are used for data reduction.

no ker reg: 87% @ 38 epochs
ker reg 0.001: 86%
no dense layers+ ker reg:
'''

from keras.models import Model
from keras import regularizers
from keras.layers import Input, Flatten, Dropout, Dense
from keras.layers import Conv2D, BatchNormalization, Activation


# wrapper function for the 2D convolution layer
def conv_layer(input, filters, kernel_size, strides=1):
      return Conv2D(filters=filters, kernel_size=kernel_size, kernel_initializer='he_uniform', \
                    kernel_regularizer=regularizers.l2(0.001), \
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
      net = conv_layer(net, 1, 5, 5)
      net = BatchNormalization()(net)
      net = Flatten()(net)
      output_layer = Activation('sigmoid')(net)


      '''
      net = conv_layer(net, 128, 1)

      net = Flatten()(net)
      net = Dropout(0.2)(net)
      output_layer = Dense(units=1, activation='sigmoid', kernel_initializer='he_uniform')(net)
      '''
      
      return Model(inputs=input_layer, outputs=output_layer)

