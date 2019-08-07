import tensorflow as tf


def unet(self,inputs,kernel_size,padding, training=False):
    with tf.name_scope('UNet'):
        if self.batch_norm:
            preactivation = tf.layers.batch_normalization(inputs,training=training)
        # Down
        res1 = tf.layers.conv2d(preactivation, filters=32, kernel_size=kernel_size, strides=2, padding=padding,activation=tf.nn.relu)
        if self.batch_norm:
            res1 = tf.layers.batch_normalization(res1,training=training)


        res2 = tf.layers.conv2d(res1, filters=64, kernel_size=kernel_size, strides=2, padding=padding,activation=tf.nn.relu)
        if self.batch_norm:
            res2 = tf.layers.batch_normalization(res2,training=training)


        res3 = tf.layers.conv2d(res2, filters=128, kernel_size=kernel_size, strides=2, padding=padding,activation=tf.nn.relu)
        if self.batch_norm:
            res3 = tf.layers.batch_normalization(res3,training=training)

        # Up
        res2_up = tf.layers.conv2d(res3, filters=64, kernel_size=kernel_size, strides=1, padding=padding,activation=tf.nn.relu)
        if self.batch_norm:
            res2_up = tf.layers.batch_normalization(res2_up,training=training)
        res2_up = tf.image.resize(res2_up, [int(self.IM_ROWS/4), int(self.IM_COLS/4)])
        res2_up = tf.concat([res2,res2_up],axis=-1)

        res1_up = tf.layers.conv2d(res2_up, filters=32, kernel_size=kernel_size, strides=1, padding=padding,activation=tf.nn.relu)
        if self.batch_norm:
            res1_up = tf.layers.batch_normalization(res1_up,training=training)
        res1_up = tf.image.resize(res1_up, [int(self.IM_ROWS / 2), int(self.IM_COLS / 2)])
        res1_up = tf.concat([res1,res1_up],axis=-1)

        recon = tf.layers.conv2d(res1_up,filters=16, kernel_size=3,strides=1,padding=padding,activation=tf.nn.relu)
        if self.batch_norm:
            recon = tf.layers.batch_normalization(recon,training=training)
        recon = tf.image.resize(recon, [self.IM_ROWS, self.IM_COLS])

        recon = tf.layers.conv2d(recon,filters=1, kernel_size=3,strides=1,padding=padding,activation=None)


    return recon, res3
