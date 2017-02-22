from __future__ import print_function

import numpy as np
import tensorflow as tf
import time

import relaax.algorithm_base.agent_base
import relaax.common.protocol.socket_protocol


def make_network(config):
    network = AgentNN(config)
    return network


class Agent(relaax.algorithm_base.agent_base.AgentBase):
    def __init__(self, config, parameter_server):
        self._config = config
        self._parameter_server = parameter_server
        self._local_network = make_network(config)

        print('Learning rate = ', self._config.learning_rate)
        print('Batch size = ', self._config.batch_size)

        initialize_all_variables = tf.variables_initializer(tf.global_variables())
        self._session = tf.Session()
        self._session.run(initialize_all_variables)

        self.gradBuffer = self._session.run(self._local_network.train_vars)
        for ix, grad in enumerate(self.gradBuffer):
            self.gradBuffer[ix] = grad * 0

        self.global_t = 0  # counter for global steps between all agents
        self.episode_reward = 0  # score accumulator for current episode (game)

        self.states = []  # auxiliary states accumulator through batch_size = 0..N
        self.actions = []  # auxiliary actions accumulator through batch_size = 0..N
        self.rewards = []  # auxiliary rewards accumulator through batch_size = 0..N

        self.episode_t = 0  # episode counter through the training
        self.avg_reward = 0

    def act(self, state):
        start = time.time()

        # Run the policy network and get an action to take
        prob = self._local_network.run_policy(self._session, state)
        action = 0 if np.random.uniform() < prob else 1

        self.states.append(state)
        self.actions.append([action])

        self.metrics().scalar('server latency', time.time() - start)

        return action

    def reward_and_act(self, reward, state):
        if self._reward(reward):
            return self.act(state)
        return None

    def reward_and_reset(self, reward):
        if not self._reward(reward):
            return None
        self.episode_t += 1

        #print("Herr =", self.episode_reward)
        score = self.episode_reward
        self.avg_reward += self.episode_reward

        self.metrics().scalar('episode reward', self.episode_reward)
        self.episode_reward = 0

        feed_dict = {
            self._local_network.s: self.states,
            self._local_network.a: self.actions,
            self._local_network.advantage: self.discounted_reward(np.vstack(self.rewards)),
        }

        grads = self._session.run(self._local_network.grads, feed_dict=feed_dict)
        for ix, grad in enumerate(grads):
            self.gradBuffer[ix] += grad

        if self.episode_t % self._config.batch_size == 0:
            self._update_global()

        self.states = []
        self.actions = []
        self.rewards = []

        return score

    def _reward(self, reward):
        self.episode_reward += reward
        self.rewards.append(reward)

        self.global_t = self._parameter_server.increment_global_t()

        return self.global_t < self._config.max_global_step

    def _update_global(self):
        avg_score = self.avg_reward / self._config.batch_size
        self.avg_reward = 0
        print("Avg reward within updates =", avg_score)

        if avg_score > 200:
            print('Training converged in {} episodes'.format(self.episode_t))
            self.global_t = self._config.max_global_step + 1

        self._session.run(self._local_network.update, feed_dict={
            self._local_network.W1_grad: self.gradBuffer[0],
            self._local_network.W2_grad: self.gradBuffer[1]
        })
        for ix, grad in enumerate(self.gradBuffer):
            self.gradBuffer[ix] = grad * 0

    def discounted_reward(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * self._config.GAMMA + r[t]
            discounted_r[t] = running_add
        # size the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r) + 1e-20
        return discounted_r

    def metrics(self):
        return self._parameter_server.metrics()


class AgentNN(object):
    def __init__(self, config):
        self._action_size = config.action_size

        self.W1 = tf.get_variable('W1', shape=[config.state_size, config.layer_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.W2 = tf.get_variable('W2', shape=[config.layer_size, self._action_size],
                                  initializer=tf.contrib.layers.xavier_initializer())

        self.s = tf.placeholder(tf.float32, [None, config.state_size])
        hidden_fc = tf.nn.relu(tf.matmul(self.s, self.W1))
        self.pi = tf.nn.sigmoid(tf.matmul(hidden_fc, self.W2))

        self.prepare_loss()
        self.prepare_grads()
        self.prepare_optimizer(config)

    def prepare_loss(self):
        self.a = tf.placeholder(tf.float32, [None, self._action_size], name="taken_action")
        self.advantage = tf.placeholder(tf.float32, name="discounted_reward")

        log_like = tf.log(self.a * (self.a - self.pi) + (1 - self.a) * (self.pi - self.a))
        self.loss = -tf.reduce_mean(log_like * self.advantage)

    def prepare_grads(self):
        self.train_vars = tf.trainable_variables()
        self.grads = tf.gradients(self.loss, self.train_vars)

    def prepare_optimizer(self, config):
        adam = tf.train.AdamOptimizer(learning_rate=config.learning_rate)

        self.W1_grad = tf.placeholder(tf.float32, name="W1_grad")
        self.W2_grad = tf.placeholder(tf.float32, name="W2_grad")
        grads = [self.W1_grad, self.W2_grad]

        self.update = adam.apply_gradients(zip(grads, self.train_vars))

    def run_policy(self, sess, s_t):
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]
