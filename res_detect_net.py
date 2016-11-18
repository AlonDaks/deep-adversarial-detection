import keras
import keras.backend as K
import numpy as np
from keras.preprocessing import image as image_utils

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from resnet50 import ResNet50
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Dense, Flatten

from PIL import Image
import os
import h5py

from keras.layers import merge
from keras.layers.core import Lambda

def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat(0, [ shape[:1]/parts, shape[1:] ])
        stride = tf.concat(0, [ shape[:1]/parts, shape[1:]*0 ])
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in xrange(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in xrange(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in xrange(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))
            
        return Model(input=model.inputs, output=merged)

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
    y = Flatten()(y)
    y = Dense(2048, activation='relu')(y)
    y = Dense(2048, activation='relu')(y)
    predictions = Dense(2, activation='softmax')(y)

    model = Model(input=base_model.input, output=predictions)

    for layer in base_model.layers:
        if type(layer) == keras.layers.normalization.BatchNormalization:
            layer.mode = 2
        layer.trainable = False

    model = make_parallel(model, 4)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return model, x


def main():
    data = h5py.File(os.path.join(FLAGS.data_dir, 'data.h5'), 'r')
    K.set_image_dim_ordering('th')
    model, x = res_detect_net()

    model.fit(data['X_train'], data['adversarial_labels_train'], shuffle='batch', batch_size=2048*4)
    model.save('/home/ubuntu/storage_volume/res_detect_net.h5')


if __name__ == '__main__':
    main()

