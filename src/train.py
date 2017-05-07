import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from srgan import SRGAN
import load

learning_rate = 1e-3
batch_size = 16
vgg_model = '../vgg19/backup/latest'

def train():
    x = tf.placeholder(tf.float32, [None, 96, 96, 3])
    is_training = tf.placeholder(tf.bool, [])

    model = SRGAN(x, is_training, batch_size)
    sess = tf.Session()
    with tf.variable_scope('srgan'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    g_train_op = opt.minimize(
        model.g_loss, global_step=global_step, var_list=model.g_variables)
    d_train_op = opt.minimize(
        model.d_loss, global_step=global_step, var_list=model.d_variables)
    init = tf.global_variables_initializer() 
    sess.run(init)

    # Restore the VGG-19 network
    var = tf.global_variables()
    vgg_var = [var_ for var_ in var if "vgg19" in var_.name]
    saver = tf.train.Saver(vgg_var)
    saver.restore(sess, vgg_model)

    # Restore the SRGAN network
    if tf.train.get_checkpoint_state('backup/'):
        saver = tf.train.Saver()
        saver.restore(sess, 'backup/latest')

    # Load the data
    x_train, x_test = load.load()

    # Train the SRGAN model
    n_iter = int(len(x_train) / batch_size)
    while True:
        epoch = int(sess.run(global_step) / n_iter / 2) + 1
        print('epoch:', epoch)
        np.random.shuffle(x_train)
        for i in tqdm(range(n_iter)):
            x_batch = normalize(x_train[i*batch_size:(i+1)*batch_size])
            sess.run(
                [g_train_op, d_train_op],
                feed_dict={x: x_batch, is_training: True})

        # Validate
        raw = normalize(x_test[:batch_size])
        mos, fake = sess.run(
            [model.downscaled, model.imitation],
            feed_dict={x: raw, is_training: False})
        save_img([mos, fake, raw], ['Input', 'Output', 'Ground Truth'], epoch)

        # Save the model
        saver = tf.train.Saver()
        saver.save(sess, 'backup/latest', write_meta_graph=False)


def save_img(imgs, label, epoch):
    for i in range(batch_size):
        fig = plt.figure()
        for j, img in enumerate(imgs):
            im = np.uint8((img[i]+1)*127.5)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            fig.add_subplot(1, len(imgs), j+1)
            plt.imshow(im)
            plt.tick_params(labelbottom='off')
            plt.tick_params(labelleft='off')
            plt.gca().get_xaxis().set_ticks_position('none')
            plt.gca().get_yaxis().set_ticks_position('none')
            plt.xlabel(label[j])
        seq_ = "{0:09d}".format(i+1)
        epoch_ = "{0:09d}".format(epoch)
        path = os.path.join('result', seq_, '{}.jpg'.format(epoch_))
        if os.path.exists(os.path.join('result', seq_)) == False:
            os.mkdir(os.path.join('result', seq_))
        plt.savefig(path)
        plt.close()


def normalize(images):
    return np.array([image/127.5-1 for image in images])


if __name__ == '__main__':
    train()

