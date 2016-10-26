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
    for i in range(start_ind, end_ind):
        image = image_utils.load_img(os.path.join(FLAGS.data_dir, 'ILSVRC2012_img_val/{0}'.format(file_names[i])))
        image = image_utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        images[i,:,:,:] = image

    images = images.astype('float32')
    X = images
    Y = np_utils.to_categorical(labels[start_ind, end_ind], FLAGS.nb_classes)

    return X, Y

def main(argv=None):
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


    num_train_images = 8000
    num_test_images = 2000
    proc_batch_size = 1000
    num_normal_batch = 850
    num_adv_batch = proc_batch_size - num_normal_batch

    f = h5py.File(os.path.join(FLAGS.storage, 'data.h5'), 'w')
    f.create_dataset('X_train', (num_train_images, 3, FLAGS.img_rows, FLAGS.img_cols), chunks=True)
    f.create_dataset('labels_train', (num_train_images, FLAGS.nb_classes), chunks=True)
    f.create_dataset('adversarial_labels_train', (num_train_images, 2), chunks=True)
    f.create_dataset('X_test', (num_test_images, 3, FLAGS.img_rows, FLAGS.img_cols), chunks=True)
    f.create_dataset('labels_test', (num_test_images, FLAGS.nb_classes), chunks=True)
    f.create_dataset('adversarial_labels_test', (num_test_images, 2), chunks=True)

    #Generate training images
    for i in np.arange(0, num_train_images, proc_batch_size):
        X, Y = data_resnet(i, i+proc_batch_size)
        X_adv = batch_eval(sess, [x], [adv_x], [X[num_normal_batch:,:,:,:]])
        X_norm = X[:num_normal_batch,:,:,:]
        f['X_train'][i:i+num_normal_batch, :]  = X_norm
        f['X_train'][i+num_normal_batch:i+proc_batch_size, :]  = X_norm

        f['labels_train'][i:i+proc_batch_size, :] = Y
        f['adversarial_labels_train'][i:i+proc_batch_size, :] = np_utils.to_categorical(np.array([0]*num_normal_batch + [1]*num_adv_batch))
        f.flush()

    print "Generated Training Data"

    #Generate test images
    for i in np.arange(0, num_test_images, proc_batch_size):
        #start at end of training data
        i += num_train_images

        X, Y = data_resnet(i, i+proc_batch_size)
        X_adv = batch_eval(sess, [x], [adv_x], [X[num_normal_batch:,:,:,:]])
        X_norm = X[:num_normal_batch,:,:,:]
        f['X_test'][i:i+num_normal_batch, :]  = X_norm
        f['X_test'][i+num_normal_batch:i+proc_batch_size, :]  = X_norm

        f['labels_test'][i:i+proc_batch_size, :] = Y
        f['adversarial_labels_test'][i:i+proc_batch_size, :] = np_utils.to_categorical(np.array([0]*num_normal_batch + [1]*num_adv_batch))
        f.flush()
        
    print "Generated Test Data"

    f.close()


    #Show Images
    # for i in range(len(normal_predictions)):
    #     if normal_predictions[i] == np.argmax(Y_test[i,:]) and adv_predictions[i] != np.argmax(Y_test[i,:]):
    #         comp = np.hstack([X_test[i,:,:,:], X_test_adv[i,:,:,:]])
    #         view_image(comp)
    #         break
    
    
def view_image(processed_image):
    print processed_image.shape
    img = Image.fromarray(np.uint8(unprocess_input(processed_image).transpose((1,2,0))))
    img.show()


if __name__ == '__main__':
    app.run()
