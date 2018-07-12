import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
import os
import shutil
from utils import *


def decoder(z,train_mode,zdim=20,reuse=None,):
    print('decoder')
    with tf.variable_scope('decoder',reuse=reuse):
           # reshape from inputs
           w_fc1 = weight_variable([zdim,64*7*7],stddev=0.02, name="w_fc1")
           b_fc1 = bias_variable([64*7*7], name="b_fc1")
           h_fc1 = tf.nn.relu(tf.matmul(z, w_fc1) + b_fc1)
           h_reshaped = tf.reshape(h_fc1, [tf.shape(z)[0], 7, 7, 64])

           W_conv_t2 = weight_variable_xavier_initialized([5, 5, 64,64], name="W_conv_t2")
           b_conv_t2 = bias_variable([64], name="b_conv_t2")
           deconv_shape = tf.stack([tf.shape(z)[0], 14, 14, 64])
           h_conv_t2 = conv2d_transpose_strided(h_reshaped, W_conv_t2, b_conv_t2,output_shape=deconv_shape)
           h_relu_t2 = tf.nn.relu(h_conv_t2)

           W_conv_t3 = weight_variable_xavier_initialized([5, 5, 32, 64], name="W_conv_t3")
           b_conv_t3 = bias_variable([32], name="b_conv_t3")
           deconv_shape = tf.stack([tf.shape(z)[0], 28, 28, 32])
           h_conv_t3 = conv2d_transpose_strided(h_relu_t2, W_conv_t3, b_conv_t3,output_shape=deconv_shape)
           h_relu_t3 = tf.nn.relu(h_conv_t3)

           W_conv1 = weight_variable_xavier_initialized([5, 5, 32, 1], name="W_conv_t4")
           b_conv1 = bias_variable([1], name="b_conv_t4")
           h_conv1 = conv2d(h_relu_t3, W_conv1,strides=[1,1,1,1]) + b_conv1
           outputs = tf.squeeze(h_conv1)

    return outputs

def encoder(data,train_mode,zdim=20,reuse=None,):
    # outputs = tf.convert_to_tensor(data)
    img = tf.expand_dims(data,axis=3)
    print('encoder')
    with tf.variable_scope('encoder', reuse=reuse):
        W_conv1 = weight_variable_xavier_initialized([5,5,1,64],name="d_w_conv1")
        b_conv1 = bias_variable([64],name="d_b_conv1")
        h_conv1 = conv2d(img, W_conv1) + b_conv1
        h_relu1 = tf.nn.relu(h_conv1)

        W_conv2 = weight_variable_xavier_initialized([5,5,64,64],name="d_w_conv2")
        b_conv2 = bias_variable([64],name="d_b_conv2")
        h_conv2 = conv2d(h_relu1, W_conv2) + b_conv2
        h_relu2 = tf.nn.relu(h_conv2)

        W_conv3 = weight_variable_xavier_initialized([5,5,64,32],name="d_w_conv3")
        b_conv3 = bias_variable([32],name="d_b_conv3")
        h_conv3 = conv2d(h_relu2, W_conv3,strides=[1,1,1,1]) + b_conv3
        h_relu3 = tf.nn.relu(h_conv3)

        batch_size = tf.shape(h_relu3)[0]
        reshape = tf.reshape(h_relu3, [batch_size, 7*7*32])

        w_fc1 = weight_variable([7*7*32, zdim], name="d_w_fc1")
        b_fc1 = bias_variable([zdim], name="d_b_fc1")
        outputs = tf.matmul(reshape, w_fc1) + b_fc1

    return outputs




def decoderFC(z,train_mode,zdim=20,reuse=None,):
    print('decoder')
    with tf.variable_scope('decoder',reuse=reuse):
           # reshape from inputs
           w_fc1 = weight_variable([zdim,128],stddev=0.02, name="w_fc1")
           b_fc1 = bias_variable([128], name="b_fc1")
           h_fc1 = tf.nn.relu(tf.matmul(z, w_fc1) + b_fc1)

           w_fc2 = weight_variable([128,256],stddev=0.02, name="w_fc2")
           b_fc2 = bias_variable([256], name="b_fc2")
           h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

           w_fc3 = weight_variable([256,1024],stddev=0.02, name="w_fc3")
           b_fc3 = bias_variable([1024], name="b_fc3")
           h_fc3 = tf.nn.relu(tf.matmul(h_fc2, w_fc3) + b_fc3)

           w_fc4 = weight_variable([1024,784],stddev=0.02, name="w_fc4")
           b_fc4 = bias_variable([784], name="b_fc4")
           h_fc4 = tf.nn.relu(tf.matmul(h_fc3, w_fc4) + b_fc4)
           outputs = tf.reshape(h_fc4,[-1,28,28])
    return outputs

def encoderFC(data,train_mode,zdim=20,reuse=None,):

    img = tf.reshape(data,[-1,784])
    print('encoder')
    with tf.variable_scope('encoder', reuse=reuse):
        w_fc4 = weight_variable([784,1024],stddev=0.02, name="w_fc1")
        b_fc4 = bias_variable([1024], name="b_fc1")
        h_fc4 = tf.nn.relu(tf.matmul(img, w_fc4) + b_fc4)

        w_fc2 = weight_variable([1024,256],stddev=0.02, name="w_fc2")
        b_fc2 = bias_variable([256], name="b_fc2")
        h_fc2 = tf.nn.relu(tf.matmul(h_fc4, w_fc2) + b_fc2)

        w_fc3 = weight_variable([256,128],stddev=0.02, name="w_fc3")
        b_fc3 = bias_variable([128], name="b_fc3")
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, w_fc3) + b_fc3)

        w_fc1 = weight_variable([128, zdim], name="d_w_fc1")
        b_fc1 = bias_variable([zdim], name="d_b_fc1")
        outputs = tf.matmul(h_fc3, w_fc1) + b_fc1

    return outputs
