import tensorflow as tf

def res_conv(inputs,filters,kernel_size,strides,padding):

    res = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=1, padding='same',activation=tf.nn.relu)
    res = tf.layers.conv2d(res, filters=filters, kernel_size=1, strides=1, padding=padding,activation=tf.nn.relu)

    # res = tf.layers.conv3d(res, filters=filters, kernel_size=kernel_size, strides=1, padding='same',activation=tf.nn.leaky_relu)
    # res = tf.layers.conv3d(res, filters=filters, kernel_size=1, strides=1, padding=padding,activation=tf.nn.leaky_relu)

    res = tf.layers.conv2d(res, filters=filters, kernel_size=kernel_size, strides=1, padding=padding,activation=tf.nn.relu)
    res = tf.layers.conv2d(res, filters=filters, kernel_size=1, strides=strides, padding=padding,activation=tf.nn.relu)

    Id = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,activation=None)
    res = res + Id

    return res