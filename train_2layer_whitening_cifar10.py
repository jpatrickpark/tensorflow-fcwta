"""
Trains a FC-WTA autoencoder on the cifar-10 dataset. Also plots
some visualizations (see --show_plots) and evaluates the learned
featurization by training an SVM on the encoded data.

Because sklearn.svm.LinearSVC is non-deterministic, the results may vary
from run to run.
"""

import os
import time
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.datasets.cifar10 import load_data

from models import FullyConnectedWTA
from util import plot_dictionary, plot_reconstruction, plot_tsne, svm_acc, timestamp, value_to_summary

from scipy.cluster.vq import whiten
import pickle

default_dir_suffix = timestamp()
try:
    with open("whichdir.txt", "a") as myfile:
        myfile.write("{}, {}: {}\n".format(sys.argv[0],sys.argv[1],default_dir_suffix))
except:
    print("Error: Provide lifetime sparsity parameter in command line.")
    print("Exiting ...")
    sys.exit()

tf.app.flags.DEFINE_string('data_dir', 'CIFAR10_whiten_28_grayscale_data/',
                           'where to load data from (or download data to)')
tf.app.flags.DEFINE_string('train_dir', 'train_%s' % default_dir_suffix,
                           'where to store checkpoints to (or load checkpoints from)')
tf.app.flags.DEFINE_string('log_dir', 'log_%s' % default_dir_suffix,
                           'where to store logs to (use with --write_logs)')
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                          'learning rate to use during training')
tf.app.flags.DEFINE_float('sparsity', float(sys.argv[1]),
                          'lifetime sparsity constraint to enforce')
tf.app.flags.DEFINE_integer('batch_size', 100,
                            'batch size to use during training')
tf.app.flags.DEFINE_integer('hidden_units', 1500,
                            'size of each ReLU (encode) layer')
tf.app.flags.DEFINE_integer('num_layers', 2,
                            'number of ReLU (encode) layers')
tf.app.flags.DEFINE_integer('train_steps', 600000,
                            'total minibatches to train')
tf.app.flags.DEFINE_integer('steps_per_display', 100,
                            'minibatches to train before printing loss')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 10000,
                            'minibatches to train before saving checkpoint')
tf.app.flags.DEFINE_integer('train_size', 50000,
                            'number of examples to use to train classifier')
tf.app.flags.DEFINE_integer('test_size', 10000,
                            'number of examples to use to test classifier')
tf.app.flags.DEFINE_boolean('use_seed', True,
                            'fix random seed to guarantee reproducibility')
tf.app.flags.DEFINE_boolean('write_logs', False,
                            'write log files')
tf.app.flags.DEFINE_boolean('show_plots', False,
                            'show visualizations')
tf.app.flags.DEFINE_integer('each_dim', 28,
                            'number of pixels in each dimension')

FLAGS = tf.app.flags.FLAGS

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])/255

def next_batch(num, data):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]

    return np.asarray(data_shuffle)

def crop_center_oneimage(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def crop_center(img_array,trim_dim):
    result = []
    for img in img_array:
        result.append(crop_center_oneimage(img,trim_dim,trim_dim))
    return np.array(result)

def whiten_images(train, test):
    # Record the indices to distinguish train and test after concatenating
    length_train = len(train)

    # Whitening according to lecture
    whole = np.concatenate((train,test))
    mean = whole.mean(axis=0, keepdims=True)
    whole = whole - mean
    whole_transpose = whole.transpose()
    u,s,v = np.linalg.svd(whole_transpose)
    whiten_matrix = u @ np.linalg.inv(np.diag(s)) @ u.transpose()
    result = whiten_matrix @ whole_transpose

    # Transpose so that I can return nice result
    result = result.transpose()

    return result[:length_train], result[length_train:], mean, whiten_matrix

def main():
    try:
        #X_train = pickle.load( open( FLAGS.data_dir+"/X_train.p", "rb" ) )
        X_train       = np.load(open( FLAGS.data_dir+"/X_train.npy", "rb"))
        y_train       = np.load(open( FLAGS.data_dir+"/y_train.npy", "rb"))
        X_test        = np.load(open( FLAGS.data_dir+"/X_test.npy", "rb"))
        y_test        = np.load(open( FLAGS.data_dir+"/y_test.npy", "rb"))
        image_mean    = np.load(open( FLAGS.data_dir+"/image_mean.npy", "rb"))
        whiten_matrix = np.load(open( FLAGS.data_dir+"/whiten_matrix.npy", "rb"))
        # add data checks here, and if wrong then error
        print("loading complete")
    except:
        print("loading failed")
        (X_train, y_train), (X_test, y_test) = load_data()
        X_train = X_train[:FLAGS.train_size]
        y_train = y_train[:FLAGS.train_size]
        X_test = X_test[:FLAGS.test_size]
        y_test = y_test[:FLAGS.test_size]

        # With fully connected network, it will be too ambitious to use 32*32 color image.
        # Reduce dimension by doing grayscale.
        X_train = rgb2gray(X_train)
        X_test = rgb2gray(X_test)

        # Crop center in order to be able to train quickly
        X_train = crop_center(X_train,FLAGS.each_dim)
        X_test = crop_center(X_test,FLAGS.each_dim)

        # Flatten arrays
        X_train = X_train.reshape(X_train.shape[0],FLAGS.each_dim**2)
        X_test = X_test.reshape(X_test.shape[0],FLAGS.each_dim**2)
        y_train = y_train.reshape(y_train.shape[0])
        y_test = y_test.reshape(y_test.shape[0])

        # Whitten images
        X_train, X_test, image_mean, whiten_matrix = whiten_images(X_train,X_test)
        if not os.path.isdir(FLAGS.data_dir):
            os.mkdir(FLAGS.data_dir)
        np.save(FLAGS.data_dir+"/X_train", X_train, True)
        np.save(FLAGS.data_dir+"/y_train", y_train, True)
        np.save(FLAGS.data_dir+"/X_test", X_test, True)
        np.save(FLAGS.data_dir+"/y_test", y_test, True)
        np.save(FLAGS.data_dir+"/image_mean", image_mean, True)
        np.save(FLAGS.data_dir+"/whiten_matrix", whiten_matrix, True)

    fcwta = FullyConnectedWTA(FLAGS.each_dim**2,
                              FLAGS.batch_size,
                              sparsity=FLAGS.sparsity,
                              hidden_units=FLAGS.hidden_units,
                              encode_layers=FLAGS.num_layers,
                              learning_rate=FLAGS.learning_rate)

    with tf.Session() as sess:
        if FLAGS.write_logs:
            writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Restoring from %s' % ckpt.model_checkpoint_path)
            fcwta.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            if not os.path.exists(FLAGS.train_dir):
                os.makedirs(FLAGS.train_dir)

        avg_time = avg_loss = 0  # Averages over FLAGS.steps_per_display steps
        step = 0
        while step < FLAGS.train_steps:
            start_time = time.time()
            batch_x = next_batch(FLAGS.batch_size, X_train)
            _, loss = fcwta.step(sess, batch_x)

            avg_time += (time.time() - start_time) / FLAGS.steps_per_display
            avg_loss += loss / FLAGS.steps_per_display
            step += 1

            if step % FLAGS.steps_per_display == 0:
                global_step = fcwta.global_step.eval()
                print('step={}, global step={}, loss={:.3f}, time={:.3f}'.format(
                    step, global_step, avg_loss, avg_time))
                if FLAGS.write_logs:
                    writer.add_summary(value_to_summary(avg_loss, 'loss'),
                                       global_step=global_step)
                avg_time = avg_loss = 0
            if step % FLAGS.steps_per_checkpoint == 0:
                checkpoint_path = FLAGS.train_dir + '/ckpt'
                fcwta.saver.save(sess,
                                 checkpoint_path,
                                 global_step=fcwta.global_step)
                print('Wrote checkpoint')

        if FLAGS.show_plots:
            # Examine code dictionary
            dictionary = fcwta.get_dictionary(sess)
            plot_dictionary(dictionary, (FLAGS.each_dim, FLAGS.each_dim), num_shown=200, row_length=20)

            # Examine reconstructions of first batch of images
            decoded, _ = fcwta.step(sess, X_train[:FLAGS.batch_size], forward_only=True)
            plot_reconstruction(X_train[:FLAGS.batch_size], decoded, (FLAGS.each_dim, FLAGS.each_dim), 20)

        # Featurize data
        X_train_f = fcwta.encode(sess, X_train)
        X_test_f = fcwta.encode(sess, X_test)

        if FLAGS.show_plots:
            # Examine t-SNE visualizations
            plot_tsne(X_train[:1000], y_train[:1000])
            plot_tsne(X_train_f[:1000], y_train[:1000])

        # Evaluate classification accuracy
        # JP 2017/11/01: This part of code is actually not tested with new data...
        for C in np.logspace(-3, 2, 6):
            acc, _ = svm_acc(X_train_f, y_train, X_test_f, y_test, C)
            print('C={:.3f}, acc={:.4f}'.format(C, acc))

if __name__ == '__main__':
    main()
