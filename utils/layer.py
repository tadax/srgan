import tensorflow as tf

def lrelu(x, trainbable=None):
    alpha = 0.2
    return tf.maximum(alpha * x, x)

def prelu(x, trainable=True):
    dim = x.get_shape()[-1]
    alpha = tf.get_variable('alpha', dim, dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=trainable)
    out = tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)
    return out

def conv_layer(x, filter_shape, stride, trainable=True):
    out_channels = filter_shape[3]
    filter_ = tf.get_variable('weight', dtype=tf.float32, shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
    out = tf.nn.conv2d(x, filter=filter_, strides=[1, stride, stride, 1], padding='SAME')
    return out

def deconv_layer(x, filter_shape, output_shape, stride, trainable=True):
    filter_ = tf.get_variable('weight', dtype=tf.float32, shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
    out = tf.nn.conv2d_transpose(x, filter=filter_, output_shape=output_shape, strides=[1, stride, stride, 1])
    return out

def max_pooling_layer(x, size, stride):
    out = tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')
    return out

def full_connection_layer(x, out_dim, trainable=True):
    input_shape = x.get_shape().as_list()
    dim = input_shape[-1]
    W = tf.get_variable('weight', dtype=tf.float32, shape=[dim, out_dim], initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
    b = tf.get_variable('bias', dtype=tf.float32, shape=[out_dim], initializer=tf.constant_initializer(0.0), trainable=trainable)
    out = tf.nn.bias_add(tf.matmul(x, W), b)
    return out

def batch_normalize(x, is_training, decay=0.99, epsilon=0.001, trainable=True):
    input_shape = x.get_shape().as_list()
    dim = input_shape[3]
    beta = tf.get_variable('beta', dtype=tf.float32, shape=[dim], initializer=tf.truncated_normal_initializer(stddev=0.0), trainable=trainable)
    scale = tf.get_variable('scale', dtype=tf.float32, shape=[dim], initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
    pop_mean = tf.get_variable('pop_mean', dtype=tf.float32, shape=[dim], initializer=tf.constant_initializer(0.0), trainable=False)
    pop_var = tf.get_variable('pop_var', dtype=tf.float32, shape=[dim], initializer=tf.constant_initializer(1.0), trainable=False)
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)
    def bn_inference():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)
    return tf.cond(is_training, bn_train, bn_inference)

def flatten_layer(x):
    input_shape = x.get_shape().as_list()
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))
    out = tf.reshape(transposed, [-1, dim])
    return out

def pixel_shuffle_layer(x, r, n_split):
    def PS(x, r):
        bs, a, b, c = x.get_shape().as_list()
        x = tf.reshape(x, (bs, a, b, r, r))
        x = tf.transpose(x, (0, 1, 2, 4, 3))
        x = tf.split(1, a, x)
        x = tf.concat(2, [tf.squeeze(x_) for x_ in x])
        x = tf.split(1, b, x)
        x = tf.concat(2, [tf.squeeze(x_) for x_ in x])
        return tf.reshape(x, (bs, a*r, b*r, 1))
    xc = tf.split(3, n_split, x)
    x = tf.concat(3, [PS(x_, r) for x_ in xc])
    return x
    
