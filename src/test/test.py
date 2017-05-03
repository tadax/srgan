import numpy as np
import scipy.misc
import cv2
import dlib
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../../utils')
sys.path.append('../../vgg19')
from srgan import SRGAN

x = tf.placeholder(tf.float32, [None, 96, 96, 3])
is_training = tf.placeholder(tf.bool, [])

model = SRGAN(x, is_training, 16)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, '../backup/latest')

img = cv2.imread('temp.jpg')
h, w = img.shape[:2]
detector = dlib.get_frontal_face_detector()
dets = detector(img, 1)
if dets is None or len(dets) != 1:
    print("unsuitable")
    exit()
d = dets[0]
if d.left() < 0 or d.top() < 0 or d.right() > w or d.bottom() > h:
    print("unsuitable")
    exit()
face = img[d.top():d.bottom(), d.left():d.right()]
face = cv2.resize(face, (96, 96))
face = face / 127.5 - 1
input_ = np.zeros((16, 96, 96, 3))
input_[0] = face

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
plt.savefig('result.jpg')

c = cv2.imread('result.jpg')
c = c[160:340, 70:585, :]
c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
scipy.misc.imsave('result.jpg', c)

