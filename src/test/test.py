import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../../utils')
sys.path.append('../../vgg19')
from srgan import SRGAN

batch_size = 16

x = tf.placeholder(tf.float32, [None, 96, 96, 3])
is_training = tf.placeholder(tf.bool, [])

model = SRGAN(x, is_training, batch_size)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, '../backup/latest')

img = cv2.imread('test.jpg')
img = img / 127.5 - 1
input_ = np.zeros((batch_size, 96, 96, 3))
input_[0] = img

mos, fake = sess.run(
    [model.downscaled, model.imitation],
    feed_dict={x: input_, is_training: False})

labels = ['Input', 'Output', 'Ground Truth']
fig = plt.figure()
for j, img in enumerate([mos[0], fake[0], input_[0]]):
    im = np.uint8((img+1)*127.5)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    fig.add_subplot(1, 3, j+1)
    plt.imshow(im)
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')
    plt.gca().get_xaxis().set_ticks_position('none')
    plt.gca().get_yaxis().set_ticks_position('none')
    plt.xlabel(labels[j])
plt.show()

