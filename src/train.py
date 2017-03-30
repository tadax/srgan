import os
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append('../utils')

from load import load
from augment import IMG
from srgan import SRGAN

learning_rate = 1e-3
batch_size = 32
img_dim = 96
IMG = IMG(normalized=True, flip=True, brightness=False, cropping=False, blur=False)

vgg_model = '../vgg19/model/backup'

def train():
    print('... building')
    model = SRGAN()
    sess = tf.Session()
    with tf.variable_scope('srgan'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.variable_scope('generator'):
        g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.g_loss, global_step=global_step, var_list=model.g_variables)
    with tf.variable_scope('discriminator'):
        d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)
    
    print('... initializing')
    sess.run(tf.global_variables_initializer())
    var_ = tf.all_variables()

    # restore VGG-19 network
    vgg_var = [var for var in var_ if "vgg19" in var.name]
    saver = tf.train.Saver(vgg_var)
    saver.restore(sess, vgg_model)

    # restore SRGAN network
    if tf.train.get_checkpoint_state('model/'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('model/')
        last_model = ckpt.model_checkpoint_path
        print('... restoring the model: {}'.format(last_model))
        saver.restore(sess, last_model)

    print('... loading')
    x_train = load('lfw/train')
    x_test = load('lfw/test')
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    print(x_train.shape)
    print(x_test.shape)

    print('... training')
    n_iter = n_train // batch_size if n_train % batch_size == 0 else n_train // batch_size + 1
    while True:
        epoch = int(sess.run(global_step) / n_iter / 2)
        print('----- epoch {} -----'.format(epoch+1))
        sum_g_loss = 0
        sum_d_loss = 0
        perm = np.random.permutation(n_train)
        x_train = x_train[perm]
        for i in tqdm(range(n_iter)):
            x_batch = x_train[i*batch_size:(i+1)*batch_size]
            x_batch = IMG.augment(x_batch)
            feed_dict = {model.x: x_batch.astype(np.float32)}
            g_optimizer.run(feed_dict=feed_dict, session=sess)
            d_optimizer.run(feed_dict=feed_dict, session=sess)
            g_loss_value = model.g_loss.eval(feed_dict=feed_dict, session=sess)
            d_loss_value = model.d_loss.eval(feed_dict=feed_dict, session=sess)
            sum_g_loss += g_loss_value
            sum_d_loss += d_loss_value
        print('G loss = {}'.format(sum_g_loss))
        print('D loss = {}'.format(sum_d_loss))

        validate(x_test, epoch, model, sess)
    
        print('... saving the model')
        saver = tf.train.Saver()
        saver.save(sess, 'model/backup', write_meta_graph=False)


def validate(x_test, epoch, model, sess):
    np.random.shuffle(x_test)
    test_data = x_test[0]
    img_h, img_w, img_c = test_data.shape
    x = np.zeros([batch_size, img_h, img_w, img_c])
    x[0] = test_data

    mos = model.downscaled.eval(feed_dict={model.x: x}, session=sess)
    mos_image = np.uint8((mos[0] + 1) * 127.5)
    mos_image = Image.fromarray(mos_image, 'RGB')
    fake = model.imitation.eval(feed_dict={model.x: x}, session=sess)
    fake_image = np.uint8((fake[0] + 1) * 127.5)
    fake_image = Image.fromarray(fake_image, 'RGB')
    real_image = np.uint8((test_data + 1) * 127.5)
    real_image = Image.fromarray(real_image, 'RGB')

    fig = plt.figure()

    fig.add_subplot(1, 3, 1)
    plt.imshow(mos_image)
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')
    plt.gca().get_xaxis().set_ticks_position('none')
    plt.gca().get_yaxis().set_ticks_position('none')
    plt.xlabel('Input')

    fig.add_subplot(1, 3, 2)
    plt.imshow(fake_image)
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')
    plt.gca().get_xaxis().set_ticks_position('none')
    plt.gca().get_yaxis().set_ticks_position('none')
    plt.xlabel('Output')

    fig.add_subplot(1, 3, 3)
    plt.imshow(real_image)
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')
    plt.gca().get_xaxis().set_ticks_position('none')
    plt.gca().get_yaxis().set_ticks_position('none')
    plt.xlabel('Ground Truth')

    i = "{0:09d}".format(epoch+1)
    path = os.path.join('result', '{}.jpg'.format(i))
    plt.savefig(path)


if __name__ == '__main__':
    train()

