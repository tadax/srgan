from tqdm import tqdm
import numpy as np
import tensorflow as tf
from vgg19 import VGG19

import sys
sys.path.append('../utils')
from augment import IMG
IMG = IMG(normalized=True, flip=True, brightness=True, cropping=True, blur=True)

learning_rate = 1e-3
batch_size = 128
img_dim = 96

def train():
    print('... building')
    model = VGG19()
    sess = tf.Session()
    with tf.variable_scope('vgg19'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.loss, global_step=global_step)
    
    print('... initializing')
    sess.run(tf.global_variables_initializer())
    if tf.train.get_checkpoint_state('model/'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('model/')
        last_model = ckpt.model_checkpoint_path
        print('... restoring the model: {}'.format(last_model))
        saver.restore(sess, last_model)

    print('... loading')
    x_train, t_train = load_data('./cifar_100/train')
    x_test, t_test = load_data('./cifar_100/test')
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    print('... training')
    n_iter = n_train / batch_size if n_train % batch_size == 0 else n_train // batch_size + 1
    while True:
        epoch = int(sess.run(global_step) / n_iter)
        print('----- epoch {} -----'.format(epoch+1))
        sum_loss = 0
        perm = np.random.permutation(n_train)
        x_train = x_train[perm]
        t_train = t_train[perm]
        for i in tqdm(range(n_iter)):
            x_batch = x_train[i*batch_size:(i+1)*batch_size]
            t_batch = t_train[i*batch_size:(i+1)*batch_size]
            x_batch = IMG.augment(n_iter)
            feed_dict = {model.x: x_batch.astype(np.float32), model.t: t_batch.astype(np.float32)}
            optimizer.run(feed_dict=feed_dict, session=sess)
            loss_value = model.loss.eval(feed_dict=feed_dict, session=sess)
            sum_loss += loss_value
        print('loss = {}'.format(sum_loss))

        train_accuracy = validate(x_train, t_train, n_train, model, sess)
        test_accuracy = validate(x_test, t_test, n_test, model, sess)
        print('train accuracy = {} %'.format(train_accuracy * 100))
        print('test accuracy = {} %'.format(test_accuracy * 100))

        print('... saving the model')
        saver = tf.train.Saver()
        saver.save(sess, 'model/backup', write_meta_graph=False)


def load():
    x = []; t = []
    paths = glob.glob(os.path.join(dir_, '*'))
    for path in paths:
        bgr_img = cv2.imread(path)
        rbg_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        img = np.array(rbg_img) / 127.5 - 1
        label = os.path.basename(path).split('_')[0]
        x.append(img)
        t.append(label)
    return (np.array(x), np.array(t))


def validate(x, t, n, model, sess):
    prediction = np.array([])
    answer = np.array([])
    for i in range(0, n, batch_size):
        x_batch = x[i:i+batch_size]
        t_batch = t[i:i+batch_size]
        x_batch = IMG.augment(x_batch)
        output = model.prob.eval(feed_dict={model.x: x_batch.astype(np.float32)}, session=sess)
        prediction = np.concatenate([prediction, np.argmax(output, 1)])
        answer = np.concatenate([answer, t_batch])
        correct_prediction = np.equal(prediction, answer)
    accuracy = np.mean(correct_prediction)
    return accuracy


if __name__ == '__main__':
    train()

