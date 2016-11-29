import keras
import keras.backend as K
import numpy as np
from keras.preprocessing import image as image_utils

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist, model_mnist
from cleverhans.utils_tf import tf_model_train, tf_model_eval, batch_eval
from cleverhans.attacks import fgsm

from keras.applications.resnet50 import ResNet50
from keras.utils import np_utils

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
flags.DEFINE_string('data_dir', '/home/ubuntu/storage_volume/data', 'location of imagenet data')
flags.DEFINE_string('storage', '/home/ubuntu/storage_volume', 'storage volume for writing dataset')

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x

def unprocess_input(x, dim_ordering="default"):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}
    if dim_ordering == 'th':
        # 'BGR'->'RGB'
        x = x[::-1, :, :]
        x[0, :, :] += 103.939
        x[1, :, :] += 116.779
        x[2, :, :] += 123.68
    else:
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x[ :, :, 0] += 103.939
        x[ :, :, 1] += 116.779
        x[ :, :, 2] += 123.68
    return x


def data_resnet(start_ind, end_ind):
    file_names = ['ILSVRC2012_val_000{0:05d}.JPEG'.format(i+1) for i in range(start_ind, end_ind)]
    labels = np.loadtxt('labels.txt')
    images = np.empty((end_ind - start_ind, 3, FLAGS.img_rows, FLAGS.img_cols))
    for i in range(end_ind - start_ind):
        image = image_utils.load_img(os.path.join(FLAGS.data_dir, file_names[i]), target_size=(224,224))
        image = image_utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        images[i,:,:,:] = image
    images = images.astype('float32')
    X = images
    Y = np_utils.to_categorical(labels[start_ind:end_ind], FLAGS.nb_classes)
    return X, Y

"""
MNIST cleverhans tutorial
:return:
"""
# Image dimensions ordering should follow the Theano convention
if keras.backend.image_dim_ordering() != 'th':
    keras.backend.set_image_dim_ordering('th')
    print "INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'tf', temporarily setting to 'th'"

# Create TF session and set as Keras backend session
sess = tf.Session()
keras.backend.set_session(sess)
print "Created TensorFlow session and set Keras backend."


# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, 3, 224, 224))
y = tf.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))


# Define TF model graph
model = ResNet50(input_tensor=x)
predictions = model.outputs[0]
print "Defined TensorFlow model graph."


# Craft adversarial examples using Fast Gradient Sign Method (FGSM)
adv_x = fgsm(x, predictions, eps=0.3)


num_train_images = 40000
num_test_images = 10000
proc_batch_size = 100 #5000
num_normal_batch = 90 #4250
num_adv_batch = proc_batch_size - num_normal_batch

f = h5py.File(os.path.join(FLAGS.storage, 'data.h5'), 'w')
f.create_dataset('X_train', (num_train_images, 3, FLAGS.img_rows, FLAGS.img_cols))
f.create_dataset('labels_train', (num_train_images, FLAGS.nb_classes))
f.create_dataset('adversarial_labels_train', (num_train_images, 2))
f.create_dataset('X_test', (num_test_images, 3, FLAGS.img_rows, FLAGS.img_cols))
f.create_dataset('labels_test', (num_test_images, FLAGS.nb_classes))
f.create_dataset('adversarial_labels_test', (num_test_images, 2))