import tensorflow as tf
import numpy as np

def PD(self, NitOut, NitIn, u, delta, alpha, lda):
    # ones = tf.ones([self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH], dtype=tf.float32, name="ones")
    #
    # td = tf.clip_by_value(tf.Variable(0.3, dtype=tf.float32, name="td"), 0.2, 0.4)
    # tp = tf.clip_by_value(tf.Variable(0.3, dtype=tf.float32, name="tp"), 0.2, 0.4)
    td = .3
    tp = .3
    sigma = 1.0
    # lda =  tf.clip_by_value( tf.reshape(par[:,0],[self.batch_size,1,1,1]),0.1,2.0)
    # alpha = tf.clip_by_value(tf.reshape(par[:,1],[self.batch_size,1,1,1]),0.5,1.5)

    p1 = tf.constant(np.zeros([self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH]),
                     dtype=tf.float32,
                     name="p1")
    p2 = tf.constant(np.zeros([self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH]),
                     dtype=tf.float32,
                     name="p2")
    # p3 = tf.constant(np.zeros([self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH, 1]),
    #                  dtype=tf.float32,
    #                  name="p3")
    # p1 = tf.stop_gradient(p1)
    # p2 = tf.stop_gradient(p2)
    # p3 = tf.stop_gradient(p3)
    # #sigma_org = sigma
    # delta_org = delta
    # alpha_org = alpha
    delta_stop = tf.stop_gradient(delta)
    alpha_stop = tf.stop_gradient(alpha)
    lda_stop = tf.stop_gradient(lda)


    u_ = u
    u_prev_out = u
    f = u

    for i in range(NitOut):

        with tf.variable_scope("OuterLoop", reuse=tf.AUTO_REUSE):
            if i == NitOut - 1:
                u_barra = (1. + tf.multiply(sigma, tf.pow(tf.div(delta, alpha), 2.))) * u - tf.multiply(
                    sigma, (
                        delta / (tf.pow(alpha, 2.)))) + 2 * (u - u_prev_out)
                u_barra_stop = tf.stop_gradient(u_barra)

                u_prev_out = u
            else:
                u_barra = (1. + tf.multiply(sigma, tf.pow(tf.div(delta_stop, alpha_stop), 2.))) * u - tf.multiply(sigma, (
                    delta_stop / (tf.pow(alpha_stop, 2.)))) + 2 * (u - u_prev_out)
                u_barra_stop = tf.stop_gradient(u_barra)

                u_prev_out = u
            for ii in range(NitIn[i]):
                if (i == NitOut-1) and (ii == NitIn[i]-1):
                    with tf.variable_scope("InnerLoop_Final", reuse=tf.AUTO_REUSE):
                        u_prev = u
                        ux, uy = self.gradient(u_)

                        p1 = p1 + tf.multiply(td, tf.multiply(alpha, ux))
                        p2 = p2 + tf.multiply(td, tf.multiply(alpha, uy))
                        mod2 = tf.pow(p1, 2.) + tf.pow(p2, 2.)
                        p_norm = tf.maximum(1., tf.sqrt(mod2 + 1e-5))

                        p1 = p1 / p_norm
                        p2 = p2 / p_norm
                        u_tilda = u + tf.multiply(tp, self.divergence(p1, p2))

                        u =  tf.maximum(0.0, tf.minimum(1., (tf.div((u_tilda + tf.multiply(tp, (
                            tf.div(lda, alpha)) * f) + tf.div(tp, sigma) * u_barra), (
                                                                      1 + tf.multiply(tp, (tf.div(lda,
                                                                                                  alpha))) + tf.div(
                                                                          tp, sigma))))))
                        u_ = 2 * u - u_prev
                else:
                    with tf.variable_scope("InnerLoop", reuse=tf.AUTO_REUSE):
                                    u_prev = u
                                    ux, uy = self.gradient(u_)

                                    p1 = p1 + tf.multiply(td, tf.multiply(alpha_stop, ux))
                                    p2 = p2 + tf.multiply(td, tf.multiply(alpha_stop, uy))
                                    mod2 = tf.pow(p1, 2.) + tf.pow(p2, 2.)
                                    p_norm = tf.maximum(1., tf.sqrt(mod2 + 1e-5))

                                    p1 = p1 / p_norm
                                    p2 = p2 / p_norm
                                    u_tilda = u + tf.multiply(tp, self.divergence(p1, p2))

                                    u = tf.maximum(0.0, tf.minimum(1., tf.div((u_tilda + tf.multiply(tp, (tf.div(lda_stop, alpha_stop)) * f) + tf.div(tp, sigma) * u_barra_stop), (1 + tf.multiply(tp, (tf.div(lda_stop,alpha_stop))) + tf.div(tp, sigma)))))
                                    # u = (tf.div((u_tilda + tf.multiply(tp, (tf.div(lda_stop, alpha_stop)) * f) + tf.div(tp, sigma) * u_barra_stop), (1 + tf.multiply(tp, (tf.div(lda_stop,alpha_stop))) + tf.div(tp, sigma))))

                                    u_ = 2 * u - u_prev


    return u, td, tp

# def PD(self, NitOut, NitIn, u, delta, alpha, lda):
#     # ones = tf.ones([self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH], dtype=tf.float32, name="ones")
#     #
#     # td = tf.clip_by_value(tf.Variable(0.3, dtype=tf.float32, name="td"), 0.2, 0.4)
#     # tp = tf.clip_by_value(tf.Variable(0.3, dtype=tf.float32, name="tp"), 0.2, 0.4)
#     td = .3
#     tp = .3
#     sigma = 1.0
#     # lda =  tf.clip_by_value( tf.reshape(par[:,0],[self.batch_size,1,1,1]),0.1,2.0)
#     # alpha = tf.clip_by_value(tf.reshape(par[:,1],[self.batch_size,1,1,1]),0.5,1.5)
#
#     p1 = tf.constant(np.zeros([self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH]),
#                      dtype=tf.float32,
#                      name="p1")
#     p2 = tf.constant(np.zeros([self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH]),
#                      dtype=tf.float32,
#                      name="p2")
#     # p3 = tf.constant(np.zeros([self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH, 1]),
#     #                  dtype=tf.float32,
#     #                  name="p3")
#     # p1 = tf.stop_gradient(p1)
#     # p2 = tf.stop_gradient(p2)
#     # p3 = tf.stop_gradient(p3)
#     # #sigma_org = sigma
#     # delta_org = delta
#     # alpha_org = alpha
#     delta_stop = tf.stop_gradient(delta)
#     alpha_stop = tf.stop_gradient(alpha)
#     lda_stop = tf.stop_gradient(lda)
#
#
#     u_ = u
#     u_prev_out = u
#     f = u
#
#     for i in range(NitOut):
#
#         with tf.variable_scope("OuterLoop", reuse=tf.AUTO_REUSE):
#             u_barra = (1. + tf.multiply(sigma, tf.pow(tf.div(delta, alpha), 2.))) * u - tf.multiply(
#                 sigma, (
#                     delta / (tf.pow(alpha, 2.)))) + 2 * (u - u_prev_out)
#             # u_barra_stop = tf.stop_gradient(u_barra)
#
#             u_prev_out = u
#
#             for ii in range(NitIn[i]):
#                 with tf.variable_scope("InnerLoop_Final", reuse=tf.AUTO_REUSE):
#                     u_prev = u
#                     ux, uy = self.gradient(u_)
#
#                     p1 = p1 + tf.multiply(td, tf.multiply(alpha, ux))
#                     p2 = p2 + tf.multiply(td, tf.multiply(alpha, uy))
#                     mod2 = tf.pow(p1, 2.) + tf.pow(p2, 2.)
#                     p_norm = tf.maximum(1., tf.sqrt(mod2 + 1e-5))
#
#                     p1 = p1 / p_norm
#                     p2 = p2 / p_norm
#                     u_tilda = u + tf.multiply(tp, self.divergence(p1, p2))
#
#                     # u = tf.maximum(0.0, tf.minimum(1., tf.div((u_tilda + tf.multiply(tp, (tf.div(lda, alpha)) * f) + tf.div(tp, sigma) * u_barra), (1 + tf.multiply(tp, (tf.div(lda,alpha))) + tf.div(tp, sigma)))))
#                     # u = tf.sigmoid(tf.div((u_tilda + tf.multiply(tp, (tf.div(lda, alpha)) * f) + tf.div(tp, sigma) * u_barra), (1 + tf.multiply(tp, (tf.div(lda,alpha))) + tf.div(tp, sigma))))
#                     u = tf.div((u_tilda + tf.multiply(tp, (tf.div(lda, alpha)) * f) + tf.div(tp, sigma) * u_barra), (1 + tf.multiply(tp, (tf.div(lda,alpha))) + tf.div(tp, sigma)))
#                     u_ = 2 * u - u_prev
#
#
#
#     return u, td, tp