import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
import os
import shutil
import matplotlib.pyplot as plt
from models import *
from utils import *


def tf_grass_proj(z,gdim):

    ztmp0 = tf.reshape(z,[-1,gdim[0],gdim[0]])
    # ztmp = tf.matmul(ztmp0,tf.transpose(ztmp0,perm=[0,2,1]))
    _,z_g,_ = tf.svd(ztmp0)
    ztmp2 = tf.matmul(z_g[:,:,:gdim[1]],tf.transpose(z_g[:,:,:gdim[1]],perm=[0,2,1]))
    z1 = tf.reshape(ztmp2,[-1,gdim[0]*gdim[0]])
    return z1

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

gdim = (5,2)

x = tf.placeholder(tf.float32, shape=[None, 28,28])
train_mode = tf.placeholder(tf.bool,name='train_mode')

z = encoder(x,train_mode,gdim[0]*gdim[0])
z1 = tf_grass_proj(z,gdim)
x_hat = decoder(z1,train_mode,gdim[0]*gdim[0])

# z_dec = encoder(x_hat,train_mode,gdim[0]*gdim[0],reuse=True)
# z2 = tf_grass_proj(z_dec,gdim)


reconstruction_loss = tf.reduce_sum(tf.square(x_hat-x))

optimizer = tf.train.AdamOptimizer(1e-3).minimize(reconstruction_loss)

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('./models_grassmann')



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("************ Model restored! **************")


    for i in range(20000):
        batch = mnist.train.next_batch(100)
        _,loss = sess.run([optimizer,reconstruction_loss],feed_dict={x:batch[0].reshape(-1,28,28),train_mode:True})
        if i%100==0:
            print('step {:d} reconstruction error: {:.2f}'.format(i,loss))
        if i %1000 ==0:
            save_path = saver.save(sess,"./models_grassmann/model_"+str(i)+".ckpt")
