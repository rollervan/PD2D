import tensorflow as tf


def unet(self,inputs,kernel_size,padding, training=False):
    with tf.name_scope('UNet'):
        if self.batch_norm:
            preactivation = tf.layers.batch_normalization(inputs,training=training)
        else:
            preactivation = inputs
        # Down
        res1 = tf.layers.conv2d(preactivation, filters=32, kernel_size=kernel_size, strides=1, padding=padding,activation=tf.nn.relu)
        if self.batch_norm:
            res1 = tf.layers.batch_normalization(res1,training=training)
        res1 = tf.layers.max_pooling2d(res1, pool_size=2, strides=2, padding=padding)


        res2 = tf.layers.conv2d(res1, filters=64, kernel_size=kernel_size, strides=1, padding=padding,activation=tf.nn.relu)
        if self.batch_norm:
            res2 = tf.layers.batch_normalization(res2,training=training)
        res2 = tf.layers.max_pooling2d(res2, pool_size=2, strides=2, padding=padding)


        res3 = tf.layers.conv2d(res2, filters=128, kernel_size=kernel_size, strides=1, padding=padding,activation=tf.nn.relu)
        if self.batch_norm:
            res3 = tf.layers.batch_normalization(res3,training=training)
        res3 = tf.layers.max_pooling2d(res3, pool_size=2, strides=2, padding=padding)

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

        recon = tf.image.resize(recon, [self.IM_ROWS, self.IM_COLS])

        recon = tf.layers.conv2d(recon,filters=1, kernel_size=3,strides=1,padding=padding,activation=None)


    return recon, res3


def real_unet(self, inputs, kernel_size, padding, training=False):
    with tf.name_scope('UNet'):
        scale = 1


        # preactivation = tf.image.resize(inputs, [120,120])
        preactivation = inputs
        pre_coded_inpunt = preactivation

        # Down
        res1 = tf.layers.conv2d(preactivation, filters=int(64*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                activation=tf.nn.relu)
        if self.batch_norm:
            res1 = tf.layers.batch_normalization(res1, training=training)
        res1 = tf.layers.conv2d(res1, filters=int(64*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                activation=tf.nn.relu)
        if self.batch_norm:
            res1 = tf.layers.batch_normalization(res1, training=training)
        res1 = tf.layers.max_pooling2d(res1, pool_size=2, strides=2, padding=padding)

        #####################

        res2 = tf.layers.conv2d(res1, filters=int(128*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                activation=tf.nn.relu)
        if self.batch_norm:
            res2 = tf.layers.batch_normalization(res2, training=training)
        res2 = tf.layers.conv2d(res2, filters=int(128*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                activation=tf.nn.relu)
        if self.batch_norm:
            res2 = tf.layers.batch_normalization(res2, training=training)
        res2 = tf.layers.max_pooling2d(res2, pool_size=2, strides=2, padding=padding)

        #####################

        res3 = tf.layers.conv2d(res2, filters=int(256*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                activation=tf.nn.relu)
        if self.batch_norm:
            res3 = tf.layers.batch_normalization(res3, training=training)
        res3 = tf.layers.conv2d(res3, filters=int(128*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                activation=tf.nn.relu)
        if self.batch_norm:
            res3 = tf.layers.batch_normalization(res3, training=training)
        res3 = tf.layers.max_pooling2d(res3, pool_size=2, strides=2, padding=padding)

        #####################

        res4 = tf.layers.conv2d(res3, filters=int(512*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                activation=tf.nn.relu)
        if self.batch_norm:
            res4 = tf.layers.batch_normalization(res4, training=training)
        res4 = tf.layers.conv2d(res4, filters=int(512*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                activation=tf.nn.relu)
        if self.batch_norm:
            res4 = tf.layers.batch_normalization(res4, training=training)
        res4 = tf.layers.max_pooling2d(res4, pool_size=2, strides=2, padding=padding)

        #####################

        res5 = tf.layers.conv2d(res4, filters=int(512*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                activation=tf.nn.relu)
        if self.batch_norm:
            res5 = tf.layers.batch_normalization(res5, training=training)
        res5 = tf.layers.conv2d(res5, filters=int(1024*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                activation=tf.nn.relu)
        if self.batch_norm:
            res5 = tf.layers.batch_normalization(res5, training=training)
        res5 = tf.layers.conv2d(res5, filters=int(1024*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                activation=tf.nn.relu)
        if self.batch_norm:
            res5 = tf.layers.batch_normalization(res5, training=training)

        # Up
        res4_up = tf.layers.conv2d(res5, filters=int(512*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                             activation=tf.nn.relu)
        if self.batch_norm:
            res4_up = tf.layers.batch_normalization(res4_up, training=training)
        res4_up = tf.concat([res4, res4_up], axis=-1)
        res4_up = tf.layers.conv2d(res4_up, filters=int(512*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                   activation=tf.nn.relu)
        if self.batch_norm:
            res4_up = tf.layers.batch_normalization(res4_up, training=training)
        res4_up = tf.layers.conv2d(res4_up, filters=int(512*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                   activation=tf.nn.relu)
        if self.batch_norm:
            res4_up = tf.layers.batch_normalization(res4_up, training=training)

        #####################

        res3_up = tf.layers.conv2d_transpose(res4_up, filters=int(256*scale), kernel_size=kernel_size, strides=2, padding=padding,
                                             activation=tf.nn.relu)
        if self.batch_norm:
            res3_up = tf.layers.batch_normalization(res3_up, training=training)
        # res3_up = res3_up[:,0:15,0:15,:]
        res3_up = tf.concat([res3, res3_up], axis=-1)
        res3_up = tf.layers.conv2d(res3_up, filters=int(256*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                   activation=tf.nn.relu)
        if self.batch_norm:
            res3_up = tf.layers.batch_normalization(res3_up, training=training)
        res3_up = tf.layers.conv2d(res3_up, filters=int(256*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                   activation=tf.nn.relu)
        if self.batch_norm:
            res3_up = tf.layers.batch_normalization(res3_up, training=training)

        #####################

        res2_up = tf.layers.conv2d_transpose(res3_up, filters=int(128*scale), kernel_size=kernel_size, strides=2, padding=padding,
                                             activation=tf.nn.relu)
        if self.batch_norm:
            res2_up = tf.layers.batch_normalization(res2_up, training=training)

        res2_up = tf.concat([res2, res2_up], axis=-1)
        res2_up = tf.layers.conv2d(res2_up, filters=int(128*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                   activation=tf.nn.relu)
        if self.batch_norm:
            res2_up = tf.layers.batch_normalization(res2_up, training=training)
        res2_up = tf.layers.conv2d(res2_up, filters=int(128*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                   activation=tf.nn.relu)
        if self.batch_norm:
            res2_up = tf.layers.batch_normalization(res2_up, training=training)

        #####################

        res1_up = tf.layers.conv2d_transpose(res2_up, filters=int(64*scale), kernel_size=kernel_size, strides=2, padding=padding,
                                             activation=tf.nn.relu)
        if self.batch_norm:
            res1_up = tf.layers.batch_normalization(res1_up, training=training)

        res1_up = tf.concat([res1, res1_up], axis=-1)
        res1_up = tf.layers.conv2d(res1_up, filters=int(64*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                   activation=tf.nn.relu)
        if self.batch_norm:
            res1_up = tf.layers.batch_normalization(res1_up, training=training)
        res1_up = tf.layers.conv2d(res1_up, filters=int(64*scale), kernel_size=kernel_size, strides=1, padding=padding,
                                   activation=tf.nn.relu)
        if self.batch_norm:
            res1_up = tf.layers.batch_normalization(res1_up, training=training)

        #####################

        recon = tf.layers.conv2d(res1_up, filters=2, kernel_size=1, strides=1, padding=padding, activation=None)

        recon = tf.image.resize(recon, [self.IM_ROWS, self.IM_COLS])

    return recon, pre_coded_inpunt


def net(self,inputs,kernel_size,padding, training=False):
    with tf.name_scope('UNet'):
        if self.batch_norm:
            preactivation = tf.layers.batch_normalization(inputs, training=training)
        else:
            preactivation = inputs
        # Down
        res1 = tf.layers.conv2d(preactivation, filters=32, kernel_size=kernel_size, strides=1, padding=padding,
                                activation=tf.nn.relu, dilation_rate=2)
        if self.batch_norm:
            res1 = tf.layers.batch_normalization(res1, training=training)
        res1 = tf.layers.max_pooling2d(res1, pool_size=2, strides=2, padding=padding)

        res2 = tf.layers.conv2d(res1, filters=64, kernel_size=kernel_size, strides=1, padding=padding,
                                activation=tf.nn.relu, dilation_rate=2)
        if self.batch_norm:
            res2 = tf.layers.batch_normalization(res2, training=training)
        res2 = tf.layers.max_pooling2d(res2, pool_size=2, strides=2, padding=padding)

        res3 = tf.layers.conv2d(res2, filters=128, kernel_size=kernel_size, strides=1, padding=padding,
                                activation=tf.nn.relu, dilation_rate=2)
        if self.batch_norm:
            res3 = tf.layers.batch_normalization(res3, training=training)
        res3 = tf.layers.max_pooling2d(res3, pool_size=2, strides=2, padding=padding)

        # Up
        res2_up = tf.layers.conv2d(res3, filters=64, kernel_size=kernel_size, strides=1, padding=padding,
                                   activation=tf.nn.relu, dilation_rate=2)
        if self.batch_norm:
            res2_up = tf.layers.batch_normalization(res2_up, training=training)
        res2_up = tf.image.resize(res2_up, [int(self.IM_ROWS / 4), int(self.IM_COLS / 4)])
        res2_up = tf.concat([res2, res2_up], axis=-1)

        res1_up = tf.layers.conv2d(res2_up, filters=32, kernel_size=kernel_size, strides=1, padding=padding,
                                   activation=tf.nn.relu)
        if self.batch_norm:
            res1_up = tf.layers.batch_normalization(res1_up, training=training)
        res1_up = tf.image.resize(res1_up, [int(self.IM_ROWS / 2), int(self.IM_COLS / 2)])
        res1_up = tf.concat([res1, res1_up], axis=-1)

        recon = tf.layers.conv2d(res1_up, filters=16, kernel_size=3, strides=1, padding=padding,
                                 activation=tf.nn.relu)

        recon = tf.image.resize(recon, [self.IM_ROWS, self.IM_COLS])

        recon = tf.layers.conv2d(recon, filters=2, kernel_size=3, strides=1, padding=padding, activation=None)

        return recon, res1
