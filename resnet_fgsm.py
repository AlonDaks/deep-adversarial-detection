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

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_classes', 1000, 'Number of classification classes')
flags.DEFINE_integer('img_rows', 224, 'Input row dimension')
flags.DEFINE_integer('img_cols', 224, 'Input column dimension')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')

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
    file_names = ['ILSVRC2012_val_00000{0:03d}.JPEG'.format(i+1) for i in range(100)]
        # 'ILSVRC2012_val_00000002.JPEG',
        # 'ILSVRC2012_val_00000003.JPEG',
        # 'ILSVRC2012_val_00000004.JPEG',
        # 'ILSVRC2012_val_00000005.JPEG',
        # 'ILSVRC2012_val_00000006.JPEG',
        # 'ILSVRC2012_val_00000007.JPEG',
        # 'ILSVRC2012_val_00000008.JPEG',
        # 'ILSVRC2012_val_00000009.JPEG',
        # 'ILSVRC2012_val_00000010.JPEG']
    labels = [490,361,171,822,297,482,13,704,599,164,649,11,73,286,554,6,648,399,749,545,13,204,318,693,399,304,102,207,480,780,644,275,14,954,249,790,501,547,809,606,297,927,424,156,60,983,256,207,281,456,413,498,561,750,182,267,118,893,597,840,836,107,647,471,945,451,214,790,291,837,707,193,397,568,401,705,200,202,31,949,361,98,709,483,563,695,122,497,914,476,102,199,104,221,138,257,188,436,229,52]

    images = np.empty((10, 3, FLAGS.img_rows, FLAGS.img_cols))
    for i in range(10):
        image = image_utils.load_img('./ILSVRC2012_img_val/{0}'.format(file_names[i]), target_size=(224, 224))
        image = image_utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        # image = image.transpose((0,3,1,2))
        images[i,:,:,:] = image
    print images.shape

    images = images.astype('float32')

    X_train, Y_train = images[:5,:,:,:], labels[:5]
    X_test, Y_test = images[5:,:,:,:], labels[5:]
    

    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    # print('X_train shape:', X_train.shape)
    # print(X_train.shape[0], 'train samples')
    # print(X_test.shape[0], 'test samples')

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
    X_train, Y_train, X_test, Y_test = data_resnet()
    print "Loaded ImageNet test data."

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 3, 224, 224))
    y = tf.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))

    # Define TF model graph
    model = ResNet50(input_tensor=x)
    predictions = model.outputs[0]
    print "Defined TensorFlow model graph."

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    accuracy = tf_model_eval(sess, x, y, predictions, X_test, Y_test)
    # assert X_test.shape[0] == 10000, X_test.shape
    print 'Test accuracy on legitimate test examples: ' + str(accuracy)

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    adv_x = fgsm(x, predictions, eps=0.3)
    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test])
    # assert X_test_adv.shape[0] == 10000, X_test_adv.shape

    # Evaluate the accuracy of the MNIST model on adversarial examples
    accuracy = tf_model_eval(sess, x, y, predictions, X_test_adv, Y_test)
    print'Test accuracy on adversarial examples: ' + str(accuracy)


if __name__ == '__main__':
    app.run()
    # data_resnet()
