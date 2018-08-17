import tensorflow as tf
#keras = tf.keras
from keras.models import Model
from keras.layers import Bidirectional, Input, Concatenate, Cropping3D, Dense, Flatten, TimeDistributed, ConvLSTM2D, LeakyReLU
from keras.layers.core import Dropout, Activation, Reshape
from keras.layers.convolutional import Conv3D, MaxPooling3D, UpSampling3D, Cropping3D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
import keras.backend as K


def conv_layer(layer, depth, size):
    conv = Conv3D(depth, size, padding='same')(layer)
    conv = LeakyReLU(0.2)(conv)
    return BatchNormalization()(conv)
    
def Generator(input_shape, output, kernel_depth, kernel_size=3):
    # 32x32x32x4
    input = Input(shape=input_shape)
    
    conv_32 = conv_layer(input, 1 * kernel_depth, kernel_size)
    pool_16 = MaxPooling3D()(conv_32)

    conv_16 = conv_layer(pool_16, 2 * kernel_depth, kernel_size)
    pool_8 = MaxPooling3D()(conv_16)

    conv_8 = conv_layer(pool_8, 4 * kernel_depth, kernel_size)

    up_16 = concatenate([UpSampling3D()(conv_8), conv_16])
    up_conv_16 = conv_layer(up_16, 2 * kernel_depth, kernel_size)

    up_32 = concatenate([UpSampling3D()(up_conv_16), conv_32])
    up_conv_32 = conv_layer(up_32, 1 * kernel_depth, kernel_size)
    
    final1 = concatenate([up_conv_32, input])
    final2 = Conv3D(output, 1, activation='softmax')(final1)
    
    model = Model(input, final2, name="Generator")
    return model
   
def Discriminator(input_shape, generator_shape, kernel_depth, kernel_size=5):
    real_input = Input(shape=input_shape)
    generator_input = Input(shape=generator_shape)    
    input = Concatenate()([real_input, generator_input])
   
    conv_32 = conv_layer(input, 1 * kernel_depth, kernel_size)
    pool_16 = MaxPooling3D()(conv_32)

    conv_16 = conv_layer(pool_16, 2 * kernel_depth, kernel_size)
    pool_8 = MaxPooling3D()(conv_16)

    conv_8 = conv_layer(pool_8, 4 * kernel_depth, kernel_size)
    pool_4 = MaxPooling3D()(conv_8)
    
    x = Flatten()(pool_4)
    x = Dense(2, activation="softmax")(x)
    
    model = Model([real_input, generator_input], x, name="Discriminator")
    return model
  
def Combine(gen, disc, input_shape, new_sequence):
    input = Input(shape=input_shape)
    generated_image = gen(input)

    reshaped = Reshape(new_sequence)(generated_image)
    
    DCGAN_output = disc([input, reshaped])

    DCGAN = Model(inputs=[input],
                  outputs=[generated_image, DCGAN_output],
                  name="Combined")

    return DCGAN
