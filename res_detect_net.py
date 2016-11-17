import keras
import keras.backend as K
import numpy as np
from keras.preprocessing import image as image_utils

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from keras.applications.resnet50 import ResNet50
from keras.utils import np_utils
from keras.layers import Dense

from PIL import Image
import os
import h5py

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_classes', 1000, 'Number of classification classes')
flags.DEFINE_integer('img_rows', 224, 'Input row dimension')
flags.DEFINE_integer('img_cols', 224, 'Input column dimension')
flags.DEFINE_integer('batch_size', 16, 'Size of training batches')
flags.DEFINE_string('data_dir', '/home/ubuntu/storage_volume', 'Size of training batches')


def res_detect_net():
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 3, 224, 224))

    base_model = ResNet50(include_top=False, input_tensor=x)
    
    y = base_model.output
    y = Dense(2048, activation='relu')(y)
    y = Dense(2048, activation='relu')(y)
    predictions = Dense(2, activation='softmax')(y)

    model = Model(input=base_model.input, output=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return model, x


def main():
    data = h5py.File(os.path.join(FLAGS.data_dir, 'data.h5'), 'r')
    K.set_image_dim_ordering('th')
    model, x, y = res_detect_net()
    
    model.fit(data['X_train'], data['adversarial_labels_train'])


if __name__ == '__main__':
    main()

