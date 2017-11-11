from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os

from layers import *
from time import time, strftime


class GenSeg:
    """GenSeg is a supervised machine learning model that semantically segments n-dimensional data.

    GenSeg is a generalized version of SegNet, in that it can operate on data with N spatial dimensions instead of
    just 2 like in the original SegNet. However, due to restrictions in TensorFlow, specifically in the way that
    convolutions work, this implementation of GenSeg only works for 1<=N<=3 spatial dimensions. This does mean that
    this implementation can handle 3D and 1D data as well as the more conventional 2D data. Additionally, this
    implementation is designed so that once support for N>3 is added into TensorFlow, it should be trivial to add
    that support into this implementation.
    """

    def __init__(self, input_shape, num_classes, seed=None, load_model=None):
        """Initializes the architecture of GenSeg and returns an instance.

        Currently, only 1<=N<=3 spatial dimensions in the input data are supported due to limitations in TensorFlow, so
        ensure that the input_shape specified follows this restriction.

        Args:
            input_shape:    A list that represents the shape of the input. Can contain None as the first element to
                            indicate that the batch size can vary (this is the preferred way to do it). Example:
                            [None, 32, 32, 32, 1] for 3D data.
            num_classes:    An integer that is equal to the number of classes that the data will be classified into.
            seed:           An integer used to seed the initial random state. Can be None to generate a new random seed.
            load_model:     If not None, then this should be a string indicating the checkpoint file containing data
                            that will be used to initialize the parameters of the model. Typically used when loading a
                            pre-trained model, or resuming a previous training session.
        """
        print("Constructing Architecture...")
        self._input_shape = tuple(input_shape)  # Tuples are used to ensure the dimensions are immutable
        x_shape = tuple(input_shape)  # 1st dim should be the size of dataset
        y_shape = tuple(input_shape[:-1])  # Rank of y should be one less
        self._num_classes = num_classes
        self._seed = seed
        self._graph = tf.Graph()
        with self._graph.as_default():
            tf.set_random_seed(seed)

            with tf.variable_scope('Input'):
                self._x = tf.placeholder(tf.float32, shape=x_shape, name="X")
                self._y = tf.placeholder(tf.int32, shape=y_shape, name="Y")
                self._phase_train = tf.placeholder(tf.bool, name="Phase")

            with tf.variable_scope('Preprocessing'):
                # We want to normalize
                x_norm, _ = batch_norm(self._x, self._phase_train, scope='X-Norm')

            with tf.variable_scope('Encoder'):
                conv1_1, _ = conv(x_norm, 64, phase_train=self._phase_train, scope='Conv1_1')
                conv1_2, _ = conv(conv1_1, 64, phase_train=self._phase_train, scope='Conv1_2')
                pool1, mask1 = pool(conv1_2, scope='Pool1')

                conv2_1, _ = conv(pool1, 128, phase_train=self._phase_train, scope='Conv2_1')
                conv2_2, _ = conv(conv2_1, 128, phase_train=self._phase_train, scope='Conv2_2')
                pool2, mask2 = pool(conv2_2, scope='Pool2')

                conv3_1, _ = conv(pool2, 256, phase_train=self._phase_train, scope='Conv3_1')
                conv3_2, _ = conv(conv3_1, 256, phase_train=self._phase_train, scope='Conv3_2')
                conv3_3, _ = conv(conv3_2, 256, phase_train=self._phase_train, scope='Conv3_3')
                pool3, mask3 = pool(conv3_3, scope='Pool3')
                drop3 = dropout(pool3, self._phase_train, scope='Drop3')

                conv4_1, _ = conv(drop3, 512, phase_train=self._phase_train, scope='Conv4_1')
                conv4_2, _ = conv(conv4_1, 512, phase_train=self._phase_train, scope='Conv4_2')
                conv4_3, _ = conv(conv4_2, 512, phase_train=self._phase_train, scope='Conv4_3')
                pool4, mask4 = pool(conv4_3, scope='Pool4')
                drop4 = dropout(pool4, self._phase_train, scope='Drop4')

            with tf.variable_scope('Decoder'):
                unpool5 = unpool(drop4, mask4, scope='Unpool5')
                conv5_1, _ = conv(unpool5, 512, phase_train=self._phase_train, scope='Conv5_1')
                conv5_2, _ = conv(conv5_1, 512, phase_train=self._phase_train, scope='Conv5_2')
                conv5_3, _ = conv(conv5_2, 512, phase_train=self._phase_train, scope='Conv5_3')
                drop5 = dropout(conv5_3, self._phase_train, scope='Drop5')

                unpool6 = unpool(drop5, mask3, scope='Unpool6')
                conv6_1, _ = conv(unpool6, 256, phase_train=self._phase_train, scope='Conv6_1')
                conv6_2, _ = conv(conv6_1, 256, phase_train=self._phase_train, scope='Conv6_2')
                conv6_3, _ = conv(conv6_2, 256, phase_train=self._phase_train, scope='Conv6_3')
                drop6 = dropout(conv6_3, self._phase_train, scope='Drop6')

                unpool7 = unpool(drop6, mask2, scope='Unpool7')
                conv7_1, _ = conv(unpool7, 128, phase_train=self._phase_train, scope='Conv7_1')
                conv7_2, _ = conv(conv7_1, 128, phase_train=self._phase_train, scope='Conv7_2')

                unpool8 = unpool(conv7_2, mask1, scope='Unpool8')
                conv8_1, _ = conv(unpool8, 256, phase_train=self._phase_train, scope='Conv8_1')
                conv8_2, _ = conv(conv8_1, 256, phase_train=self._phase_train, scope='Conv8_2')

            with tf.variable_scope('Softmax'):
                scores, _ = conv(conv8_2, num_classes, phase_train=None, size=1, scope='Scores')
                self._y_hat = tf.nn.softmax(scores, name='Y-Hat')  # Operates on last dimension

            with tf.variable_scope('Pipelining'):
                self._loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=self._y),
                    name='Loss'
                )
                self._train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self._loss)

            self._sess = tf.Session(graph=self._graph)  # Not sure if this really needs to explicitly specify the graph
            with self._sess.as_default():
                self._saver = tf.train.Saver()
                if load_model is not None:
                    print("Restoring Model...")
                    load_model = os.path.abspath(load_model)
                    self._saver.restore(self._sess, load_model)
                    print("Model Restored!")
                else:
                    print("Initializing model...")
                    self._sess.run(tf.global_variables_initializer())
                    print("Model Initialized!")

    def train(self, x_train, y_train, num_epochs, start_stop_info=True, progress_info=True):
        """Trains the model using the data provided as a batch.

        Because GenSeg typically runs on large datasets, it is often infeasible to load the entire dataset on either
        memory or the GPU. For this reason, the selection of batches is left up to the user, so that s/he can load the
        proper number of data. For best performance, try to make the batch size (size of first dimension) as large as
        possible without exceeding memory to take advantage of the vectorized code that TensorFlow uses.

        That being said, if the entire dataset fits in memory (and GPU memory if using GPU) and mini-batching is not
        desired, then it is preferable to pass the whole dataset to this function and use a higher value for num_epochs.

        Args:
            x_train:  A numpy ndarray that contains the data to train over. Should should have a shape of
                [batch_size, spatial_dim1, ... , spatial_dimN, channels]. Only 1<=N<=3 spatial dimensions are supported
                currently. These should correspond to the shape of y_train.

            y_train:  A numpy ndarray that contains the labels that correspond to the data being trained on. Should have
                a shape of [batch_size, spatial_dim1, ... , spatial_dimN]. Only 1<=N<=3 spatial dimensions are supported
                currently. These should correspond to the shape of x_train. The actual values in the tensor should be
                integers corresponding to the labels. There should only be num_classes unique integers.

            num_epochs:  The number of iterations over the provided batch to perform until training is considered to be
                complete. If all your data fits in memory and you don't need to mini-batch, then this should be a large
                number (>1000). Otherwise keep this small (<50) so the model doesn't become skewed by the small size of
                the provided mini-batch too quickly.

            start_stop_info:  If true, print when the training begins and ends.

            progress_info:  If true, print what the current loss and percent completion over the course of training.

        Returns:
            The loss value after training
        """
        with self._sess.as_default():
            # Training loop for parameter tuning
            if start_stop_info:
                print("Starting training for %d epochs" % num_epochs)
            last_time = time()
            for epoch in range(num_epochs):
                _, loss_val = self._sess.run(
                    [self._train_step, self._loss],
                    feed_dict={self._x: x_train, self._y: y_train, self._phase_train: True}
                )
                current_time = time()
                if progress_info and (current_time - last_time) >= 5:  # Only print progress every 5 seconds
                    last_time = current_time
                    print("Current Loss Value: %.10f, Percent Complete: %.4f" % (loss_val, epoch / num_epochs * 100))
            if start_stop_info:
                print("Completed Training.")
            return loss_val

    def apply(self, x_data):
        """Applies the model to the batch of data provided. Typically called after the model is trained.

        Args:
            x_data:  A numpy ndarray of the data to apply the model to. Should have the same shape as the training data.
                Example: x_data.shape is [batch_size, num_features0, 480, 3] for a 640x480 RGB image

        Returns:
            A numpy ndarray of the data, with the last dimension being the class probabilities instead of channels.
            Example: result.shape is [batch_size, 640, 480, 10] for a 640x480 RGB image with 10 target classes
        """
        with self._sess.as_default():
            return self._sess.run(self._y_hat, feed_dict={self._x: x_data, self._phase_train: False})

    def save_model(self, save_path=None):
        """Saves the model in the specified file.

        Args:
            save_path:  The relative path to the file. By default, it is
                saved/GenSeg-Year-Month-Date_Hour-Minute-Second.ckpt
        """
        with self._sess.as_default():
            print("Saving Model")
            if save_path is None:
                save_path = "saved/GenSeg-%s.ckpt" % strftime("%Y-%m-%d_%H-%M-%S")
            dirname = os.path.dirname(save_path)
            if dirname is not '':
                os.makedirs(dirname, exist_ok=True)
            save_path = os.path.abspath(save_path)
            path = self._saver.save(self._sess, save_path)
            print("Model successfully saved in file: %s" % path)
