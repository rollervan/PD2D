import tensorflow as tf

# def res_conv(self,inputs,filters,kernel_size,strides,padding, training=False):
#     with tf.name_scope('Res_module'):
#
#         res = tf.layers.conv2d(inputs, filters=filters, kernel_size=1, strides=1, padding=padding,activation=tf.nn.relu)
#         if self.batch_norm:
#             res = tf.layers.batch_normalization(res,training=training)
#
#         res = tf.layers.conv2d(res, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,activation=tf.nn.relu)
#         if self.batch_norm:
#             res = tf.layers.batch_normalization(res,training=training)
#
#         res1 = tf.layers.conv2d(res, filters=filters, kernel_size=1, strides=1, padding=padding,activation=tf.nn.relu)
#         if self.batch_norm:
#             res1 = tf.layers.batch_normalization(res1,training=training)
#
#
#         res = tf.layers.conv2d(inputs, filters=filters, kernel_size=1, strides=1, padding=padding,activation=tf.nn.relu)
#         if self.batch_norm:
#             res = tf.layers.batch_normalization(res,training=training)
#
#         res = tf.layers.conv2d(res, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,activation=tf.nn.relu)
#         if self.batch_norm:
#             res = tf.layers.batch_normalization(res,training=training)
#
#         res2 = tf.layers.conv2d(res, filters=filters, kernel_size=1, strides=1, padding=padding,activation=tf.nn.relu)
#         if self.batch_norm:
#             res2 = tf.layers.batch_normalization(res2,training=training)
#
#
#
#         res = tf.concat([res1,res2],axis=-1)
#         res = tf.layers.conv2d(res, filters=filters, kernel_size=kernel_size, strides=1, padding=padding,activation=tf.nn.relu)
#         if self.batch_norm:
#             res = tf.layers.batch_normalization(res,training=training)
#
#
#         Id = tf.layers.conv2d(inputs, filters=filters, kernel_size=1, strides=strides, padding=padding,activation=tf.nn.relu)
#         res = res + Id
#
#     return res


# def res_conv(self,inputs,filters,kernel_size,strides,padding, training=False):
#     with tf.name_scope('Res_module'):
#
#         res = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=1, padding=padding,activation=tf.nn.relu)
#         if self.batch_norm:
#             res = tf.layers.batch_normalization(res,training=training)
#
#         res = tf.layers.conv2d(res, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,activation=tf.nn.relu)
#         if self.batch_norm:
#             res = tf.layers.batch_normalization(res,training=training)
#
#         Id = tf.layers.conv2d(inputs, filters=filters, kernel_size=1, strides=strides, padding=padding,activation=None)
#         res = res + Id
#
#     return res

def res_conv(self,inputs,filters,kernel_size,strides,padding, training=False):
    with tf.name_scope('Res_module'):

        res = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=1, padding=padding,activation=tf.nn.relu)
        if self.batch_norm:
            res = tf.layers.batch_normalization(res,training=training)

        res = tf.layers.conv2d(res, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,activation=tf.nn.relu)
        if self.batch_norm:
            res = tf.layers.batch_normalization(res,training=training)

    return res
