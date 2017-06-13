'''MobileNet models for Keras.
# Reference
- [MobileNets: Efficient Convolutional Neural Networks for
   Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf))
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K

from depthwise_conv import DepthwiseConvolution2D

TH_WEIGHTS_PATH = 'https://github.com/titu1994/MobileNetworks/releases/download/v1.0/mobilenet_th_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH = 'https://github.com/titu1994/MobileNetworks/releases/download/v1.0/mobilenet_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/titu1994/MobileNetworks/releases/download/v1.0/mobilenet_th_dim_ordering_tf_kernels_no_top.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/titu1994/MobileNetworks/releases/download/v1.0/mobilenet_tf_dim_ordering_tf_kernels_no_top.h5'


def MobileNets(input_shape=None, alpha=1, depth_multiplier=1,
               include_top=True, weights='imagenet',
               input_tensor=None, classes=1001):
    ''' Instantiate the MobileNet architecture.
        Note that only TensorFlow is supported for now,
        therefore it only works with the data format
        `image_data_format='channels_last'` in your Keras config
        at ~/.keras/keras.json.
        # Arguments
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `channels_last` data format)
                or (3, 224, 224) (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 32.
                E.g. `(200, 200, 3)` would be one valid value.
            alpha: width multiplier of the MobileNet.
            depth_multiplier: depth multiplier for depthwise convolution
                (also called the resolution multiplier)
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: `None` (random initialization) or
                `imagenet` (ImageNet weights)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Returns
            A Keras model instance.
        '''

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1001:
        raise ValueError('If using `weights` as ImageNet with `include_top`'
                         ' as true, `classes` should be 1001')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      include_top=False)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_mobilenet(classes, img_input, include_top, alpha, depth_multiplier)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='mobilenet')

    # load weights
    if weights == 'imagenet':
        if (alpha == 1) and (depth_multiplier == 1):
            if K.backend() == 'theano':
                raise AttributeError('Weights for Theano backend are not available, '
                                     'as Theano does not support depthwise convolution yet.')

            # Default parameters match. Weights for this model exist:
            if K.image_data_format() == 'channels_first':
                if include_top:
                    weights_path = get_file('mobilenet_th_dim_ordering_tf_kernels.h5',
                                            TH_WEIGHTS_PATH,
                                            cache_subdir='models')
                else:
                    weights_path = get_file('mobilenet_th_dim_ordering_tf_kernels_no_top.h5',
                                            TH_WEIGHTS_PATH_NO_TOP,
                                            cache_subdir='models')

                model.load_weights(weights_path)
            else:
                if include_top:
                    weights_path = get_file('mobilenet_tf_dim_ordering_tf_kernels.h5',
                                            TF_WEIGHTS_PATH,
                                            cache_subdir='models')
                else:
                    weights_path = get_file('mobilenet_tf_dim_ordering_tf_kernels_no_top.h5',
                                            TF_WEIGHTS_PATH_NO_TOP,
                                            cache_subdir='models')

                model.load_weights(weights_path)
    return model


def __conv_block(input, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)

    x = Convolution2D(filters, kernel, padding='same', use_bias=True, strides=strides,
                      name='conv1')(input)
    x = BatchNormalization(axis=channel_axis, scale=False, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)

    return x


def __depthwise_conv_block(input, depthwise_conv_filters, pointwise_conv_filters,
                           alpha, depth_multiplier=1, strides=(1, 1), id=1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    depthwise_conv_filters = int(depthwise_conv_filters * alpha)
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConvolution2D(depthwise_conv_filters, (3, 3), padding='same', depth_multiplier=depth_multiplier,
                               strides=strides, use_bias=True, name='conv_dw_%d' % id)(input)
    x = BatchNormalization(axis=channel_axis, scale=False, name='conv_dw_%d_bn' % id)(x)
    x = Activation('relu', name='conv_dw_%d_relu' % id)(x)

    x = Convolution2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=True, strides=(1, 1),
                      name='conv_pw_%d' % id)(x)
    x = BatchNormalization(axis=channel_axis, scale=False, name='conv_pw_%d_bn' % id)(x)
    x = Activation('relu', name='conv_pw_%d_relu' % id)(x)

    return x


def __create_mobilenet(classes, img_input, include_top, alpha, depth_multiplier):
    ''' Creates a MobileNet model with specified parameters
    Args:
        classes: Number of output classes
        img_input: Input tensor or layer
        include_top: Flag to include the last dense layer
        alpha: width multiplier of the MobileNet.
        depth_multiplier: depth multiplier for depthwise convolution
                          (also called the resolution multiplier)
    Returns: a Keras Model
    '''

    x = __conv_block(img_input, 32, alpha, strides=(2, 2))
    x = __depthwise_conv_block(x, 32, 64, alpha, depth_multiplier, id=1)

    x = __depthwise_conv_block(x, 64, 128, alpha, depth_multiplier, strides=(2, 2), id=2)
    x = __depthwise_conv_block(x, 128, 128, alpha, depth_multiplier, id=3)

    x = __depthwise_conv_block(x, 128, 256, alpha, depth_multiplier, strides=(2, 2), id=4)
    x = __depthwise_conv_block(x, 256, 256, alpha, depth_multiplier, id=5)

    x = __depthwise_conv_block(x, 256, 512, alpha, depth_multiplier, strides=(2, 2), id=6)
    x = __depthwise_conv_block(x, 512, 512, alpha, depth_multiplier, id=7)
    x = __depthwise_conv_block(x, 512, 512, alpha, depth_multiplier, id=8)
    x = __depthwise_conv_block(x, 512, 512, alpha, depth_multiplier, id=9)
    x = __depthwise_conv_block(x, 512, 512, alpha, depth_multiplier, id=10)
    x = __depthwise_conv_block(x, 512, 512, alpha, depth_multiplier, id=11)

    x = __depthwise_conv_block(x, 512, 1024, alpha, depth_multiplier, strides=(2, 2), id=12)
    x = __depthwise_conv_block(x, 1024, 1024, alpha, depth_multiplier, id=13)

    x = GlobalAveragePooling2D()(x)

    if include_top:
        x = Dense(classes, activation='softmax')(x)

    return x

if __name__ == "__main__":
    from keras import backend as K
    K.set_image_data_format('channels_first')
    model = MobileNets(alpha=1, depth_multiplier=1, weights=None)

    model.summary()
