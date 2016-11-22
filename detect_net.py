import keras
import keras.backend as K
import numpy as np
from keras.preprocessing import image as image_utils

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from resnet50 import ResNet50
from keras.models import Model, load_model
from keras.utils import np_utils
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization

from PIL import Image
import os
import h5py

from keras.layers import merge
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint

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


def alex_detect_net(mode=2):

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 3, 224, 224))

    img_input = x

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    y = ZeroPadding2D((3, 3))(img_input)
    y = Convolution2D(96, 11, 11, subsample=(4, 4), name='conv1')(y)
    y = BatchNormalization(axis=bn_axis, name='bn_conv1', mode=mode)(y)
    y = Activation('relu')(y)
    y = MaxPooling2D((3, 3), strides=(2, 2))(y)

    y = Convolution2D(256, 5, 5, name='conv2')(y)
    y = BatchNormalization(axis=bn_axis, name='bn_conv2', mode=mode)(y)
    y = Activation('relu')(y)
    y = MaxPooling2D((3, 3), strides=(2, 2))(y)

    y = Convolution2D(384, 3, 3, name='conv3')(y)
    y = BatchNormalization(axis=bn_axis, name='bn_conv3', mode=mode)(y)
    y = Activation('relu')(y)

    y = Convolution2D(384, 3, 3, name='conv4')(y)
    y = BatchNormalization(axis=bn_axis, name='bn_conv4', mode=mode)(y)
    y = Activation('relu')(y)

    y = Convolution2D(384, 3, 3, name='conv5')(y)
    y = BatchNormalization(axis=bn_axis, name='bn_conv5', mode=mode)(y)
    y = Activation('relu')(y)

    y = MaxPooling2D((3, 3), strides=(2, 2))(y)

    y = Flatten()(y)
    y = Dense(1024, activation='relu')(y)
    y = Dense(1024, activation='relu')(y)
    predictions = Dense(2, activation='softmax')(y)

    model = Model(img_input, output=predictions)

    # model = make_parallel(model, 4)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model, x 


def train_res_detect_net():
    data = h5py.File(os.path.join(FLAGS.data_dir, 'data.h5'), 'r')
    K.set_image_dim_ordering('th')
    model, x = res_detect_net()

    checkpointer = ModelCheckpoint(filepath="/home/ubuntu/storage_volume/res_detect_net/weights.{epoch:02d}.hdf5", verbose=1)
    model.fit(data['X_train'], data['adversarial_labels_train'], shuffle='batch', batch_size=128, callbacks=[checkpointer])
    model.save('/home/ubuntu/storage_volume/res_detect_net/res_detect_net.h5')


def train_alex_detect_net():
    data = h5py.File(os.path.join(FLAGS.data_dir, 'data.h5'), 'r')
    K.set_image_dim_ordering('th')
    model, x = alex_detect_net()

    checkpointer = ModelCheckpoint(filepath="/home/ubuntu/storage_volume/alex_detect_net/weights.{epoch:02d}.hdf5", verbose=1)
    model.fit(data['X_train'], data['adversarial_labels_train'], shuffle='batch', batch_size=128*4, callbacks=[checkpointer])
    model.save('/home/ubuntu/storage_volume/alex_detect_net/alex_detect_net.h5')


def main(model='alex_detect_net'):
    if model == 'alex_detect_net':
        train_alex_detect_net()
    elif model == 'res_detect_net':
        train_res_detect_net()
    else:
        print 'Error: Could not train. Model type {0} unknown'.format(model)
    

def validate():
    data = h5py.File(os.path.join(FLAGS.data_dir, 'data.h5'), 'r')
    K.set_image_dim_ordering('th')
    # model = load_model('/home/ubuntu/storage_volume/res_detect_net/res_detect_net.h5')
    model, _ = res_detect_net()
    model.load_weights('/home/ubuntu/storage_volume/res_detect_net/res_detect_net.h5')
    indices = np.random.permutation(range(10000))[:1000]
    predicted_values = model.predict(data['X_test'][indices], batch_size=128, verbose=1)
    np.savetxt('pred.txt', predicted_values)
    true_values = data['adversarial_labels_test'][indices]
    print np.mean(data['adversarial_labels_test'] == predicted_values)


if __name__ == '__main__':
    main()
    # validate()

