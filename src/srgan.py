import tensorflow as tf
import numpy as np

import sys
sys.path.append('../utils')
sys.path.append('../vgg19')
from layer import *
from vgg19 import VGG19
vgg = VGG19(False)

class SRGAN:
    def __init__(self):
        self.K = 4
        self.img_dim = 96
        self.batch_size = 32
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.img_dim, self.img_dim, 3])
        self.is_training = tf.placeholder(tf.bool)
        self.downscaled = self.downscale(self.x)
        self.imitation = self.generator(self.downscaled, self.is_training, False)
        self.true_output = self.discriminator(self.x, self.is_training, False)
        self.fake_output = self.discriminator(self.imitation, self.is_training, True)
        self.g_loss, self.d_loss = self.inference_losses(self.x, self.imitation, self.true_output, self.fake_output)

    def generator(self, x, is_training, reuse):
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('deconv1'):
                x = deconv_layer(x, [3, 3, 64, 3], [self.batch_size, 24, 24, 64], 1, 'deconv1')
            x = tf.nn.relu(x)
            shortcut = x
            for i in range(5):
                mid = x
                with tf.variable_scope('block{}a'.format(i+1)):
                    x = deconv_layer(x, [3, 3, 64, 64], [self.batch_size, 24, 24, 64], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('block{}b'.format(i+1)):
                    x = deconv_layer(x, [3, 3, 64, 64], [self.batch_size, 24, 24, 64], 1)
                    x = batch_normalize(x, is_training)
                x = tf.add(x, mid)
            with tf.variable_scope('deconv2'):
                x = deconv_layer(x, [3, 3, 64, 64], [self.batch_size, 24, 24, 64], 1)
                x = batch_normalize(x, is_training)
                x = tf.add(x, shortcut)
            with tf.variable_scope('deconv3'):
                x = deconv_layer(x, [3, 3, 256, 64], [self.batch_size, 24, 24, 256], 1)
                x = pixel_shuffle_layer(x, 2, 64) # n_split = 256 / 2 ** 2
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv4'):
                x = deconv_layer(x, [3, 3, 64, 64], [self.batch_size, 48, 48, 64], 1)
                x = pixel_shuffle_layer(x, 2, 16)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv5'):
                x = deconv_layer(x, [3, 3, 3, 16], [self.batch_size, 96, 96, 3], 1)

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return x


    def discriminator(self, x, is_training, reuse):
        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1'):
                x = conv_layer(x, [3, 3, 3, 64], 1)
                x = lrelu(x)
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 64], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv3'):
                x = conv_layer(x, [3, 3, 64, 128], 1)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv4'):
                x = conv_layer(x, [3, 3, 128, 128], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv5'):
                x = conv_layer(x, [3, 3, 128, 256], 1)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv6'):
                x = conv_layer(x, [3, 3, 256, 256], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv7'):
                x = conv_layer(x, [3, 3, 256, 512], 1)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv8'):
                x = conv_layer(x, [3, 3, 512, 512], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            x = flatten_layer(x)
            with tf.variable_scope('fc'):
                x = full_connection_layer(x, 1024)
                x = lrelu(x)
            with tf.variable_scope('softmax'):
                x = full_connection_layer(x, 1)
                
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return x


    def downscale(self, x):
        K = self.K
        arr = np.zeros([K, K, 3, 3])
        arr[:, :, 0, 0] = 1.0 / K ** 2
        arr[:, :, 1, 1] = 1.0 / K ** 2
        arr[:, :, 2, 2] = 1.0 / K ** 2
        weight = tf.constant(arr, dtype=tf.float32)
        downscaled = tf.nn.conv2d(x, weight, strides=[1, K, K, 1], padding='SAME')
        return downscaled


    def inference_losses(self, x, imitation, true_output, fake_output):
        def inference_content_loss(x, imitation):
            _, x_phi = vgg.build_model(x, tf.constant(False), True)
            _, imitation_phi = vgg.build_model(imitation, tf.constant(False), True)
            content_loss = None
            for i in range(len(x_phi)):
                if content_loss is None:
                    content_loss = tf.nn.l2_loss(x_phi[i] - imitation_phi[i])
                else:
                    content_loss = content_loss + tf.nn.l2_loss(x_phi[i] - imitation_phi[i])
            return tf.reduce_mean(content_loss)

        def inference_adversarial_loss(true_output, fake_output):
            alpha = 1e-5
            g_loss = tf.reduce_mean(tf.nn.l2_loss(fake_output - tf.ones_like(fake_output)))
            d_loss_real = tf.reduce_mean(tf.nn.l2_loss(true_output - tf.ones_like(true_output)))
            d_loss_fake = tf.reduce_mean(tf.nn.l2_loss(fake_output + tf.ones_like(fake_output)))
            d_loss = d_loss_real + d_loss_fake
            return (g_loss * alpha, d_loss * alpha)

        content_loss = inference_content_loss(x, imitation)
        generator_loss, discriminator_loss = inference_adversarial_loss(true_output, fake_output)
        g_loss = content_loss + generator_loss
        d_loss = discriminator_loss
        return (g_loss, d_loss)

