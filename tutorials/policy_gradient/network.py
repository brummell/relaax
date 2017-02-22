import tensorflow as tf


class AgentNN(object):
    def __init__(self, config):
        self._action_size = config.action_size

        self.global_t = tf.Variable(0, tf.int64)
        self.increment_global_t = tf.assign_add(self.global_t, 1)

        self.W1 = tf.get_variable('W1', shape=[config.state_size, config.layer_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.W2 = tf.get_variable('W2', shape=[config.layer_size, self._action_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.values = [self.W1, self.W2]
        self.first = True

        self.s = tf.placeholder(tf.float32, [None, config.state_size])
        hidden_fc = tf.nn.relu(tf.matmul(self.s, self.W1))
        self.pi = tf.nn.sigmoid(tf.matmul(hidden_fc, self.W2))

        self._placeholders = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]
        self._assign_values = tf.group(*[
            tf.assign(v, p) for v, p in zip(self.values, self._placeholders)
            ])

        self.prepare_loss()
        self.compute_grads(config)
        self.apply_grads(config)

    def prepare_loss(self):
        self.a = tf.placeholder(tf.float32, [None, self._action_size], name="taken_action")
        self.advantage = tf.placeholder(tf.float32, name="discounted_reward")

        log_like = tf.log(self.a * (self.a - self.pi) + (1 - self.a) * (self.pi - self.a))
        self.loss = -tf.reduce_mean(log_like * self.advantage)

    def compute_grads(self, config):
        if self.first:
            self.grads = tf.gradients(self.loss, self.values)
        else:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
            self.grads = self.optimizer.compute_gradients(self.loss, self.values)

    def apply_grads(self, config):
        if self.first:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)

        self.gradients = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]

        self.update = self.optimizer.apply_gradients(zip(self.gradients, self.values))

    def run_policy(self, sess, s_t):
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]

    def assign_values(self, session, values):
        session.run(self._assign_values, feed_dict={
            p: v for p, v in zip(self._placeholders, values)
        })
