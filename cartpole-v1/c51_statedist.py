# Source from https://github.com/sorryformyself/tensorflow2_cartpole/blob/master/categorical/categorical.py
import tensorflow as tf
import gym
import numpy as np
import tensorflow.keras as keras
from sklearn.cluster import KMeans
import time
import random
from collections import deque
import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')
episodes = 1000
step_limit = 200
memory_size = 100000
env = gym.make('CartPole-v1')
env.seed(777)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

saveFileName = 'categorical_statedist'


class Network(tf.keras.Model):
    def __init__(self, support, atom_size):
        super(Network, self).__init__()

        self.support = support
        self.atom_size = atom_size

        self.fc1 = tf.keras.layers.Dense(128)
        self.fc2 = tf.keras.layers.Dense(128)
        self.advantage_output = tf.keras.layers.Dense(atom_size * action_size)
        self.value_out = tf.keras.layers.Dense(1 * atom_size)
        self.norm_advantage_output = tf.keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))
        self.build((None, state_size))

    @tf.function
    def call(self, input_tensor):
        dist = self.dist(input_tensor)
        x = tf.reduce_sum(dist * self.support, axis=2)
        return x

    @tf.function
    def dist(self, x):
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        y = self.advantage_output(x)
        y = self.norm_advantage_output(y)
        y = tf.reshape(y, (-1, action_size, self.atom_size))
        z = self.value_out(x)
        z = tf.reshape(z, (-1, 1, self.atom_size))
        x = y + z
        dist = tf.nn.softmax(x, axis=-1)
        return dist


class DQNAgent:
    def __init__(self):
        # other hyperparameters
        self.save_graph = True
        self.isTraining = True
        self.keepTraining = False
        self.play = False
        self.render = False
        self.save_model = True
        self.load_model = False
        self.random = False
        # epsilon greedy exploration
        self.initial_epsilon = 1.0
        self.epsilon = self.initial_epsilon
        self.min_epsilon = 0.01
        self.linear_annealed = (self.initial_epsilon - self.min_epsilon) / 2000
        self.decay_rate = 0.995

        # check the hyperparameters
        if self.random == True:
            self.play = False
            self.isTraining = False
        if self.play == True:
            self.render = True
            self.save_model = False
            self.load_model = True
            self.isTraining = False
            self.keepTraining = False
        if self.keepTraining == True:
            self.epsilon = self.min_epsilon
            self.load_model = True
        # fixed q value - two networks
        self.learning_rate = 0.0001
        self.fixed_q_value_steps = 100
        self.target_network_counter = 0

        # experience replay
        self.batch_size = 64
        self.gamma = 0.9
        self.replay_start_size = 320
        self.experience_replay = deque(maxlen=memory_size)
        self.kmeans = None

        # categorical DQN
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.v_min = 0.0
        self.v_max = 200.0
        self.atom_size = 51
        self.support = np.linspace(
            self.v_min, self.v_max, self.atom_size
        )
        self.delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
        if self.load_model:
            self.model = keras.models.load_model(saveFileName + '.h5')
            self.target_model = keras.models.load_model(saveFileName + '.h5')
        else:
            self.model = Network(self.support, self.atom_size)
            self.target_model = Network(self.support, self.atom_size)
            # self.target_model.predict(np.zeros((1,state_size)))

    # these methods:sample,store are used in experience replay
    def sample(self, n, re_eval=False):
        """
           Re-evaluate kmeans if re_eval=True"
        """
        n_clusters = 64
        states = np.vstack([data[0] for data in self.experience_replay])
        if re_eval:
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(states)
            labels = self.kmeans.labels_
        else:
            labels = self.kmeans.predict(states)
        h = dict()
        for c in range(n_clusters):
            h[c] = (labels == c).sum()
        counts = np.array([h[c] for c in labels])
        probabilities = 1./ (n_clusters * counts)
        return random.choices(self.experience_replay, weights=probabilities, k=n)

    def store(self, experience):
        self.experience_replay.append((state, action, reward, next_state, done))

    def create_model(self):
        inputs = tf.keras.Input(shape=(state_size,))
        fc1 = tf.keras.layers.Dense(128, activation='relu')(inputs)
        fc2 = tf.keras.layers.Dense(128, activation='relu')(fc1)
        advantage_output = tf.keras.layers.Dense(action_size, activation='linear')(fc2)
        model = tf.keras.Model(inputs, advantage_output)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss='mse',
                      metrics=['accuracy'])
        return model

    def training(self, re_eval):
        if len(self.experience_replay) >= self.replay_start_size:
            batches = self.sample(self.batch_size, re_eval or len(self.experience_replay)==self.replay_start_size)
            states = np.vstack([data[0] for data in batches])
            actions = np.vstack([data[1] for data in batches])
            rewards = np.vstack([data[2] for data in batches])
            next_states = np.vstack([data[3] for data in batches])
            dones = np.vstack([data[4] for data in batches])
            self.train_body(states, actions, rewards, next_states, dones)

    @tf.function
    def train_body(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            elementwise_loss = self._compute_td_error_body(states, actions, rewards, next_states, dones)
            absolute_errors = tf.abs(elementwise_loss)
            loss = tf.reduce_mean(elementwise_loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return absolute_errors

    @tf.function
    def _compute_td_error_body(self, states, actions, rewards, next_states, dones):
        # DDQN double DQN: choose action first in current network,
        # no axis=1 will only have one value
        max_action_next = tf.argmax(self.model(next_states), axis=1)
        batch_size_range = tf.expand_dims(tf.range(self.batch_size, dtype=tf.int64), axis=1)  # (batch_size, 1)
        max_action_next = tf.expand_dims(max_action_next, axis=1)
        next_indexes = tf.concat(values=(batch_size_range, max_action_next), axis=1)  # (batch_size, 2)
        next_dist = self.target_model.dist(next_states)
        next_dist = tf.gather_nd(next_dist, next_indexes)

        t_z = tf.where(dones, rewards, rewards + self.gamma * self.support)  # (batch_size, )
        t_z = tf.clip_by_value(t_z, self.v_min, self.v_max)
        b = tf.cast((t_z - self.v_min) / self.delta_z, tf.float32)
        l = tf.cast(tf.math.floor(b), tf.int32)
        u = tf.cast(tf.math.ceil(b), tf.int32)

        offset = tf.tile(tf.cast(tf.linspace(0., (self.batch_size - 1.) * self.atom_size, self.batch_size
                                             ), tf.int32)[:, None], (1, self.atom_size))
        proj_dist = tf.zeros(next_dist.shape, dtype=tf.float32)
        loffset = tf.reshape((l + offset), (-1,))
        uoffset = tf.reshape((u + offset), (-1,))
        u_next = tf.reshape((next_dist * (tf.cast(u, tf.float32) - b)), (-1,))
        l_next = tf.reshape((next_dist * (b - tf.cast(l, tf.float32))), (-1,))

        proj_dist = tf.add(tf.reshape(proj_dist, (-1,)), tf.gather(u_next, loffset))
        proj_dist = tf.add(tf.reshape(proj_dist, (-1,)), tf.gather(l_next, uoffset))
        proj_dist = tf.reshape(proj_dist, (self.batch_size, self.atom_size))

        dist = self.model.dist(states)
        indexes = tf.concat(values=(batch_size_range, actions), axis=1)  # (batch_size, 2)
        log_p = tf.math.log(tf.gather_nd(dist, indexes))
        loss = tf.reduce_sum(-(proj_dist * log_p), axis=1)
        return loss

    def acting(self, state):
        if self.render:
            env.render()
        self.target_network_counter += 1
        if self.target_network_counter % self.fixed_q_value_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.linear_annealed
        if np.random.sample() <= self.epsilon:
            return np.random.randint(action_size)

        action = self._get_action_body(state).numpy()
        return action

    @tf.function
    def _get_action_body(self, state):
        qvalues = self.model(state)[0]
        return tf.argmax(qvalues)


num_trials = 3
plot_rewards = []
for k in range(num_trials):
    print(f'trial {k}')
    agent = DQNAgent()
    episode_rewards = []
    if agent.isTraining:
        scores_window = deque(maxlen=100)
        start = time.time()
        for episode in range(1, episodes + 1):
            rewards = 0
            state = env.reset()
            state = np.array([state])
            steps = 0
            while True:
                action = agent.acting(state)
                next_state, reward, done, _ = env.step(action)
                rewards += reward
                next_state = next_state[None, :]
                reward = -10 if done else reward
                agent.store((state, action, reward, next_state, done))
                state = next_state
                if episode < 200:
                    interval = 20
                elif episode < 600:
                    interval = 50
                else:
                    interval = 100
                agent.training(steps % interval == 0)
                steps += 1
                if done or steps >= step_limit:
                    episode_rewards.append(rewards)
                    scores_window.append(rewards)

                    break
            print('\rEpisode {}\tAverage Score: {:.2f}\tepsilon:{:.2f}'.format(episode,
                                                                                                 np.mean(scores_window),
                                                                                                 agent.epsilon), end="")

            if np.mean(scores_window) > 195:
                print("\nproblem solved in {} episode with {:.2f} seconds".format(episode, time.time() - start))
                #agent.model.save(saveFileName + '.h5')
                #break
            if episode % 100 == 0:
                print("\nRunning for {:.2f} seconds".format(time.time() - start))
    plot_rewards.append(np.array(episode_rewards))

plt.figure(figsize=(20,5))
plt.plot(np.arange(episodes), np.mean(plot_rewards, axis=0))
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('C51_statedist.png')
print('plotted')

import pickle
with open('c51_statedist.pkl','wb') as f:
    pickle.dump(np.mean(plot_rewards, axis=0), f)
env.close()

