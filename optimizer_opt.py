# coding=utf-8
import tensorflow as tf

def optimizer_opt(self, loss, var_list, global_step, show_gradients=False):
    optim = tf.train.AdamOptimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        if show_gradients:

            grads = optim(self.learning_rate).compute_gradients(loss, var_list=var_list)
            grads_wo_none = []
            for grad, var in grads:
                if grad is not None:
                    grads_wo_none.append((grad, var))
                else:
                    print(var)
            # grads = grads_wo_none

            optimizer = optim(self.learning_rate).apply_gradients(grads_and_vars=grads, global_step=global_step)
            tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '_gradient', grad)
                else:
                    print(grad)
                # tf.summary.histogram(var.op.name + '_gradient', grad)
        else:
            optimizer = optim(self.learning_rate).minimize(loss, var_list=var_list, global_step=global_step)

    return optimizer
