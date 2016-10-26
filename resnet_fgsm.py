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

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_classes', 1000, 'Number of classification classes')
flags.DEFINE_integer('img_rows', 224, 'Input row dimension')
flags.DEFINE_integer('img_cols', 224, 'Input column dimension')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_string('data_dir', '/storage-volume/data', 'Size of training batches')

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


def data_resnet():
    file_names = ['ILSVRC2012_val_000{0:05d}.JPEG'.format(i+1) for i in range(50000)]
    labels = np.loadtxt('labels.txt')
    images = np.empty((50000, 3, FLAGS.img_rows, FLAGS.img_cols))
    for i in range(50000):
        image = image_utils.load_img(os.path.join(FLAGS.data_dir, 'ILSVRC2012_img_val/{0}'.format(file_names[i])))
        image = image_utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        images[i,:,:,:] = image

    images = images.astype('float32')

    X_train, Y_train = images[:40000,:,:,:], labels[:40000]
    X_test, Y_test = images[40000:,:,:,:], labels[40000:]

    # # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train, FLAGS.nb_classes)
    Y_test = np_utils.to_categorical(Y_test, FLAGS.nb_classes)

    return X_train, Y_train, X_test, Y_test

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

    # Get MNIST test data
    X_train_split, Y_train_split, X_test_split, Y_test_split = data_resnet()
    print "Loaded ImageNet test data."


    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 3, 224, 224))
    y = tf.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))


    # Define TF model graph
    model = ResNet50(input_tensor=x)
    predictions = model.outputs[0]
    print "Defined TensorFlow model graph."


    # Evaluate the accuracy of the MNIST model on legitimate test examples
    # accuracy = tf_model_eval(sess, x, y, predictions, X_test, Y_test)
    # print 'Test accuracy on legitimate test examples: ' + str(accuracy)
    # normal_predictions = sess.run(tf.argmax(predictions, 1), feed_dict={x: X_test, keras.backend.learning_phase(): 1})


    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    adv_x = fgsm(x, predictions, eps=0.3)
    
    X_train_normal= X_train_split[:34000,:,:,:]
    X_train_adv, = batch_eval(sess, [x], [adv_x], [X_train_split[34000:,:,:,:]])
    X_train = np.concatenate(X_train_normal, X_train_adv, axis=0)
    assert X_train.shape == X_train_split.shape
    np.save('X_train.npy', X_train)
    np.save('labels_train.npy', Y_train_split)
    np.save('adv_train.npy', np.array([0]*34000 + [1]*16000))
    print "Generated Training Data"

    X_test_normal= X_test_split[:8500,:,:,:]
    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test_split[8500:,:,:,:]])
    X_test = np.concatenate(X_test_normal, X_test_adv, axis=0)
    assert X_test.shape == X_test_split.shape
    np.save('X_test.npy', X_test)
    np.save('labels_test.npy', Y_test_split)
    np.save('adv_test.npy', np.array([0]*8500 + [1]*1500))
    print "Generated Test Data"


    # adv_predictions = sess.run(tf.argmax(predictions, 1), feed_dict={x: X_test_adv, keras.backend.learning_phase(): 1})

    #Show Images
    # for i in range(len(normal_predictions)):
    #     if normal_predictions[i] == np.argmax(Y_test[i,:]) and adv_predictions[i] != np.argmax(Y_test[i,:]):
    #         comp = np.hstack([X_test[i,:,:,:], X_test_adv[i,:,:,:]])
    #         view_image(comp)
    #         break
    
    

    # Evaluate the accuracy of the MNIST model on adversarial examples
    # accuracy = tf_model_eval(sess, x, y, predictions, X_test_adv, Y_test)
    # print'Test accuracy on adversarial examples: ' + str(accuracy)

def view_image(processed_image):
    print processed_image.shape
    img = Image.fromarray(np.uint8(unprocess_input(processed_image).transpose((1,2,0))))
    img.show()


if __name__ == '__main__':
    app.run()
