import keras
import numpy as np
from keras.preprocessing import image as image_utils

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist, model_mnist
from cleverhans.utils_tf import tf_model_train, tf_model_eval, batch_eval
from cleverhans.attacks import fgsm

from keras.applications.resnet50 import ResNet50

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_classes', 1000, 'Number of classification classes')
flags.DEFINE_integer('img_rows', 224, 'Input row dimension')
flags.DEFINE_integer('img_cols', 224, 'Input column dimension')


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


def data_resnet():
    file_names = ['ILSVRC2012_val_00000001.JPEG',
        'ILSVRC2012_val_00000002.JPEG',
        'ILSVRC2012_val_00000003.JPEG',
        'ILSVRC2012_val_00000004.JPEG',
        'ILSVRC2012_val_00000005.JPEG',
        'ILSVRC2012_val_00000006.JPEG',
        'ILSVRC2012_val_00000007.JPEG',
        'ILSVRC2012_val_00000008.JPEG',
        'ILSVRC2012_val_00000009.JPEG',
        'ILSVRC2012_val_00000010.JPEG']
    labels = [490, 361, 171, 822, 297, 482, 13, 704, 599, 164]

    images = np.array()
    for f in file_names:
        image = image_utils.load_img('./ILSVRC2012_img_val/{0}'.format(f), target_size=(224, 224))
        image = image_utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        images = np.append(images, image, axis=0)
    print images.shape


    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    # print('X_train shape:', X_train.shape)
    # print(X_train.shape[0], 'train samples')
    # print(X_test.shape[0], 'test samples')

    # # convert class vectors to binary class matrices
    # Y_train = np_utils.to_categorical(y_train, FLAGS.nb_classes)
    # Y_test = np_utils.to_categorical(y_test, FLAGS.nb_classes)
    # return X_train, Y_train, X_test, Y_test

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
    X_train, Y_train, X_test, Y_test = data_resnet()
    print "Loaded MNIST test data."

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 3, 224, 224))
    y = tf.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))

    # Define TF model graph
    model = ResNet50(input_tensor=x)
    predictions = model(x)
    print "Defined TensorFlow model graph."

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    accuracy = tf_model_eval(sess, x, y, predictions, X_test, Y_test)
    assert X_test.shape[0] == 10000, X_test.shape
    print 'Test accuracy on legitimate test examples: ' + str(accuracy)

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    adv_x = fgsm(x, predictions, eps=0.3)
    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test])
    assert X_test_adv.shape[0] == 10000, X_test_adv.shape

    # Evaluate the accuracy of the MNIST model on adversarial examples
    accuracy = tf_model_eval(sess, x, y, predictions, X_test_adv, Y_test)
    print'Test accuracy on adversarial examples: ' + str(accuracy)

    print "Repeating the process, using adversarial training"
    # Redefine TF model graph
    model_2 = model_mnist()
    predictions_2 = model_2(x)
    adv_x_2 = fgsm(x, predictions_2, eps=0.3)
    predictions_2_adv = model_2(adv_x_2)

    # Perform adversarial training
    tf_model_train(sess, x, y, predictions_2, X_train, Y_train, predictions_adv=predictions_2_adv)

    # Evaluate the accuracy of the adversarialy trained MNIST model on
    # legitimate test examples
    accuracy = tf_model_eval(sess, x, y, predictions_2, X_test, Y_test)
    print 'Test accuracy on legitimate test examples: ' + str(accuracy)

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM) on
    # the new model, which was trained using adversarial training
    X_test_adv_2, = batch_eval(sess, [x], [adv_x_2], [X_test])
    assert X_test_adv_2.shape[0] == 10000, X_test_adv_2.shape

    # Evaluate the accuracy of the adversarially trained MNIST model on
    # adversarial examples
    accuracy_adv = tf_model_eval(sess, x, y, predictions_2, X_test_adv_2, Y_test)
    print'Test accuracy on adversarial examples: ' + str(accuracy_adv)

# if __name__ == '__main__':
#     app.run()
