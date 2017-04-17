import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from load import load
from srgan import SRGAN

learning_rate = 1e-4
batch_size = 32
img_dim = 96

vgg_model = '../vgg19/backup/latest'

def train():
    model = SRGAN()
    sess = tf.Session()
    with tf.variable_scope('srgan'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
    g_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.g_loss, global_step=global_step, var_list=model.g_variables)
    d_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)
    
    sess.run(tf.global_variables_initializer())
    var_ = tf.global_variables()

    # Restore the VGG-19 network
    vgg_var = [var for var in var_ if "vgg19" in var.name]
    saver = tf.train.Saver(vgg_var)
    saver.restore(sess, vgg_model)

    # Restore the SRGAN network
    if tf.train.get_checkpoint_state('backup/'):
        saver = tf.train.Saver()
        saver.restore(sess, 'backup/latest')

    # Loading the data
    x_train = load('lfw/train')
    x_test = load('lfw/test')

    # Train the SRGAN model
    n_iter = int(np.ceil(len(x_train) / batch_size))
    while True:
        epoch = int(sess.run(global_step) / n_iter / 2)
        print('epoch: {}'.format(epoch + 1))
        perm = np.random.permutation(len(x_train))
        x_train = x_train[perm]
        for i in tqdm(range(n_iter)):
            x_batch = x_train[i * batch_size:(i + 1) * batch_size]
            sess.run([g_train_op, d_train_op], feed_dict={model.x: x_batch, model.is_training: True})

        # Validate
        validate(x_test, epoch, model, sess)

        # Save the model
        saver = tf.train.Saver()
        saver.save(sess, 'backup/latest', write_meta_graph=False)


def validate(x_test, epoch, model, sess):
    raw = x_test[:batch_size]
    mos, fake = sess.run([model.downscaled, model.imitation], feed_dict={model.x: raw, model.is_training: False})
    saveimg([mos, fake, raw], ['Input', 'Output', 'Ground Truth'], epoch)


def saveimg(imgs, label, epoch):
    for i in range(batch_size):
        fig = plt.figure()
        for j, img in enumerate(imgs):
            im = np.uint8((img[i] + 1) * 127.5)
            fig.add_subplot(1, len(imgs), j+1)
            plt.imshow(im)
            plt.tick_params(labelbottom='off')
            plt.tick_params(labelleft='off')
            plt.gca().get_xaxis().set_ticks_position('none')
            plt.gca().get_yaxis().set_ticks_position('none')
            plt.xlabel(label[j])
        seq_ = "{0:09d}".format(i + 1)
        epoch_ = "{0:09d}".format(epoch + 1)
        path = os.path.join('result', seq_, '{}.jpg'.format(epoch_))
        if os.path.exists(os.path.join('result', seq_)) == False:
            os.mkdir(os.path.join('result', seq_))
        plt.savefig(path)
        plt.close()

if __name__ == '__main__':
    train()

