import tensorflow as tf
import numpy as np


# Simple 2-layer fully-connected Policy Neural Network
class GlobalPolicyNN(object):
    # This class is used for global-NN and holds only weights on which applies computed gradients
    def __init__(self, config):
        self.global_t = tf.Variable(0, tf.int64)
        self.increment_global_t = tf.assign_add(self.global_t, 1)

        self._RMSP_DECAY = config.RMSP_DECAY
        self._RMSP_EPSILON = config.RMSP_EPSILON

        self.W1 = tf.get_variable('W1', shape=[config.state_size, config.layer_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.W2 = tf.get_variable('W2', shape=[config.layer_size, 1],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.values = [
            self.W1, self.W2
        ]

        self._placeholders = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]
        self._assign_values = tf.group(*[
            tf.assign(v, p) for v, p in zip(self.values, self._placeholders)
            ])

        self.gradients = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]
        self.learning_rate = tf.placeholder(tf.float32)

    def assign_values(self, session, values):
        session.run(self._assign_values, feed_dict={
            p: v for p, v in zip(self._placeholders, values)
            })

    def get_vars(self):
        return self.values

    def apply_gradients(self):
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate,
            decay=self._RMSP_DECAY,
            epsilon=self._RMSP_EPSILON
        )
        self.apply_gradients = optimizer.apply_gradients(zip(self.gradients, self.values))
        return self


class AgentPolicyNN(GlobalPolicyNN):
    # This class additionally implements loss computation and gradients wrt this loss
    def __init__(self, config):
        super(AgentPolicyNN, self).__init__(config)

        # state (input)
        self.s = tf.placeholder(tf.float32, [None] + config.state_size)

        hidden_fc = tf.nn.relu(tf.matmul(self.s, self.W1))

        # policy (output)
        self.pi = tf.nn.sigmoid(tf.matmul(hidden_fc, self.W2))

    def run_policy(self, sess, s_t):
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]

    def compute_gradients(self):
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate,
            decay=self._RMSP_DECAY,
            epsilon=self._RMSP_EPSILON
        )
        grads_and_vars = optimizer.compute_gradients(self.loss, self.values)
        self.grads = [grad for grad, _ in grads_and_vars]
        return self

    def prepare_loss(self, config):
        # taken action
        self.y = tf.placeholder(tf.float32, [None, config.action_size])

        # R (input for value)
        self.r = tf.placeholder("float", [None])

        log_prob = self.y - self.pi

        self.loss = None

        return self