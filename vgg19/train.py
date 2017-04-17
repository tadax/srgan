from tqdm import tqdm
import numpy as np
import tensorflow as tf
from vgg19 import VGG19

import sys
sys.path.append('../utils')
from load import load
from augment import IMG
IMG = IMG(normalized=True, flip=True, brightness=False, cropping=False, blur=False)

learning_rate = 1e-4
batch_size = 128
img_dim = 96

def train():
    model = VGG19(True)
    sess = tf.Session()
    with tf.variable_scope('vgg19'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(model.loss, global_step=global_step)
    init = tf.global_variables_initializer()
    sess.run(init)

    if tf.train.get_checkpoint_state('backup/'):
        saver = tf.train.Saver()
        saver.restore(sess, 'backup/latest')

    x_train, t_train = load('./cifar_100/train')
    x_test, t_test = load('./cifar_100/test')

    n_iter = int(np.ceil(len(x_train) / batch_size))
    while True:
        epoch = int(sess.run(global_step) / n_iter)
        print('----- epoch {} -----'.format(epoch+1))
        sum_loss = 0
        perm = np.random.permutation(len(x_train))
        x_train = x_train[perm]
        t_train = t_train[perm]
        for i in tqdm(range(0, len(x_train), batch_size)):
            x_batch = IMG.augment(x_train[i:i+batch_size])
            t_batch = t_train[i:i+batch_size]
            _, loss_value = sess.run([train_op, model.loss], feed_dict={model.x: x_batch, model.t: t_batch, model.is_training: True})
            sum_loss += loss_value
        print('loss: {}'.format(sum_loss))

        saver = tf.train.Saver()
        saver.save(sess, 'backup/latest', write_meta_graph=False)

        test_accuracy = validate(x_test, t_test, model, sess)
        print('accuracy: {} %'.format(test_accuracy * 100))


def validate(x, t, model, sess):
    prediction = np.array([])
    answer = np.array([])
    for i in range(0, len(x), batch_size):
        x_batch = IMG.augment(x[i:i+batch_size])
        t_batch = t[i:i+batch_size]
        output = model.out.eval(feed_dict={model.x: x_batch, model.is_training: False}, session=sess)
        prediction = np.concatenate([prediction, np.argmax(output, 1)])
        answer = np.concatenate([answer, t_batch])
        correct_prediction = np.equal(prediction, answer)
    accuracy = np.mean(correct_prediction)
    return accuracy


if __name__ == '__main__':
    train()

