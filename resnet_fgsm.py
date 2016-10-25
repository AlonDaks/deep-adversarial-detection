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
    file_names = ['ILSVRC2012_val_00000{0:03d}.JPEG'.format(i+1) for i in range(100)]
    labels = [65,970,230,809,516,57,334,415,674,332,109,286,370,757,595,147,108,23,478,517,334,173,948,727,23,846,270,167,55,858,324,573,150,981,586,887,32,398,777,74,516,756,129,198,256,725,565,167,717,394,92,29,844,591,358,468,259,994,872,588,474,183,107,46,842,390,101,887,870,841,467,149,21,476,80,424,159,275,175,461,970,160,788,58,479,498,369,28,487,50,270,383,366,780,373,705,330,142,949,349]
    images = np.empty((100, 3, FLAGS.img_rows, FLAGS.img_cols))
    for i in range(100):
        image = image_utils.load_img('./ILSVRC2012_img_val/{0}'.format(file_names[i]), target_size=(224, 224))
        image = image_utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)        
        images[i,:,:,:] = image

    images = images.astype('float32')

    X_train, Y_train = images[:5,:,:,:], labels[:5]
    X_test, Y_test = images[50:,:,:,:], labels[50:]
    

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
    # accuracy = tf_model_eval(sess, x, y, predictions, X_test, Y_test)
    # print 'Test accuracy on legitimate test examples: ' + str(accuracy)
    normal_predictions = sess.run(tf.argmax(predictions, 1), feed_dict={x: X_test, keras.backend.learning_phase(): 1})


    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    adv_x = fgsm(x, predictions, eps=0.3)
    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test])

    adv_predictions = sess.run(tf.argmax(predictions, 1), feed_dict={x: X_test_adv, keras.backend.learning_phase(): 1})

    for i in range(len(normal_predictions)):
        if normal_predictions[i] == np.argmax(Y_test[i,:]) and adv_predictions[i] != np.argmax(Y_test[i,:]):
            comp = np.hstack([X_test[i,:,:,:], X_test_adv[i,:,:,:]])
            view_image(comp)
            break
    else:
        print("No adversarial examples.")
    
    

    # Evaluate the accuracy of the MNIST model on adversarial examples
    # accuracy = tf_model_eval(sess, x, y, predictions, X_test_adv, Y_test)
    # print'Test accuracy on adversarial examples: ' + str(accuracy)

def view_image(processed_image):
    print processed_image.shape
    img = Image.fromarray(np.uint8(unprocess_input(processed_image).transpose((1,2,0))))
    img.show()


if __name__ == '__main__':
    app.run()
