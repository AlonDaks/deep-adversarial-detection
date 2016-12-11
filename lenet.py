import keras
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adagrad, Adam

import os

from keras.layers import merge
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

from cleverhans.attacks import fgsm
from cleverhans.utils_tf import tf_model_train, tf_model_eval, batch_eval

flags.DEFINE_integer('batch_size', 256, 'Size of training batches')


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

def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    n_train, height, width = X_train.shape
    n_test = X_test.shape[0]

    X_train = X_train.reshape(n_train, 1, height, width).astype('float32')
    X_test = X_test.reshape(n_test, 1, height, width).astype('float32')

    X_train /= 255
    X_test /= 255

    n_classes = 10

    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)

    return X_train, y_train, X_test, y_test

def lenet_mnist():
    model = Sequential()
    model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape=(1, 28, 28)))
    model.add(MaxPooling2D((2, 2), strides=(1,1)))
    model.add(Convolution2D(16, 5, 5))
    model.add(MaxPooling2D((2, 2), strides=(1,1)))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def generate_adversarial():
    keras.backend.set_image_dim_ordering('th')
    sess = tf.Session()
    keras.backend.set_session(sess)

    model = load_model('weights/lenet_trained_weights.h5')
    X_train_raw, y_train, X_test_raw, y_test = get_data()

    x = tf.placeholder(tf.float32, shape=(None, 1, 28, 28))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    x = model.inputs[0]
    predictions = model.outputs[0]

    # epsilons = [25, 10, 5, 1, 0.5]
    epsilons = [0.05]

    def get_random_halves(n):
        p = np.random.permutation(np.arange(n))
        return p[:n//2], p[n//2:]

    for epsilon in epsilons:
        print "Generating data for epsilon = {0}".format(epsilon)
        adv_x = fgsm(x, predictions, eps=epsilon)
        
        train_adv_inds, train_normal_inds = get_random_halves(X_train_raw.shape[0])
        test_adv_inds, test_normal_inds = get_random_halves(X_test_raw.shape[0])

        X_train = np.empty(X_train_raw.shape)
        X_test = np.empty(X_test_raw.shape)
        
        X_train_adv = batch_eval(sess, [x], [adv_x], [X_train_raw[train_adv_inds,:,:,:]])
        X_test_adv = batch_eval(sess, [x], [adv_x], [X_train_raw[test_adv_inds,:,:,:]])

        X_train[train_adv_inds,:,:,:] = X_train_adv
        X_test[test_adv_inds,:,:,:] = X_test_adv

        X_train[train_normal_inds,:,:,:] = X_train_raw[train_normal_inds,:,:,:]
        X_test[test_normal_inds,:,:,:] = X_test_raw[test_normal_inds,:,:,:]

        y_train_adv = np.ones((y_train.shape[0], )).astype('int32')
        y_train_adv[train_normal_inds] = 0

        y_test_adv = np.ones((y_test.shape[0], )).astype('int32')
        y_test_adv[test_normal_inds] = 0

        y_train_adv = to_categorical(y_train_adv, 2)
        y_test_adv = to_categorical(y_test_adv, 2)

        np.savez('data/mnist_modified_eps_{0}.npz'.format(epsilon), 
            X_train = X_train,
            X_test = X_test,
            y_train_adv = y_train_adv,
            y_test_adv = y_test_adv,
            y_train = y_train,
            y_test = y_test)

        print model.evaluate(X_train, y_train)

def get_le_detect_model(filename=None):
    if filename == None:
        model = Sequential()
        model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape=(1, 28, 28)))
        model.add(MaxPooling2D((2,2), strides=(1,1)))
        model.add(Flatten())
        model.add(Dense(50))
        model.add(Dense(50))
        model.add(Dense(2, activation='softmax'))
    else:
        model = load_model(filename)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_le_detect_net():
    keras.backend.set_image_dim_ordering('th')

    data = np.load('data/mnist_modified_eps_0.05.npz')

    model = get_le_detect_model()
    checkpointer = ModelCheckpoint(filepath="lenet_weights_epoch.{epoch:02d}.hdf5", verbose=1)
    model.fit(data['X_train'], data['y_train_adv'], callbacks=[checkpointer])
    model.save('lenet_trained_weights.h5')

    results = model.evaluate(data['X_test'], data['y_test_adv'])
    print results

def adversarial_le_detect_net():
    keras.backend.set_image_dim_ordering('th')
    sess = tf.Session()
    keras.backend.set_session(sess)

    data = np.load('data/mnist_modified_eps_0.05.npz')
    model = load_model('weights/eps005/lenet_trained_weights.h5')

    print model.evaluate(data['X_test'], data['y_test_adv'])

    x = model.inputs[0]
    predictions = model.outputs[0]
    adv_x = fgsm(x, predictions, eps=10)

    X_adv = batch_eval(sess, [x], [adv_x], [data['X_test']])

    print np.sum(data['X_test'] - X_adv)

    print model.evaluate(X_adv, data['y_test_adv'])

    return data['X_test'], X_adv, data['y_test_adv']






def main():
    keras.backend.set_image_dim_ordering('th')
    model = lenet_mnist()
    X_train, y_train, _, _ = get_data()
    checkpointer = ModelCheckpoint(filepath="lenet_weights_epoch.{epoch:02d}.hdf5", verbose=1)
    model.fit(X_train, y_train, callbacks=[checkpointer])
    model.save('lenet_trained_weights.h5')




if __name__ == '__main__':
    # generate_adversarial()
    # train_le_detect_net()
    X, adv, y = adversarial_le_detect_net()
    np.save('adversarial.txt', adv)



