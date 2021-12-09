# Source from https://github.com/sorryformyself/tensorflow2_cartpole/blob/master/categorical/categorical.py
import tensorflow as tf
import gym
import numpy as np
import tensorflow.keras as keras
import time
import random
from collections import deque
import matplotlib.pyplot as plt
from utils.random_dict import RandomDict

tf.get_logger().setLevel('ERROR')
episodes = 1000
step_limit = 200
memory_size = 100000
env = gym.make('CartPole-v1')
env.seed(777)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

saveFileName = 'dqn_ser'


def create_model(lr):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(state_size,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(action_size, activation='relu')]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse', metrics=['accuracy'])
    return model

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

        # n-step learning
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)

        # experience replay
        self.batch_size = 64
        self.gamma = 0.9
        self.replay_start_size = 320
        self.experience_replay = deque(maxlen=memory_size)
        self.pair_to_indices_dict = RandomDict()
        self.size_now = 0
        self.i = 0

        if self.load_model:
            self.model = keras.models.load_model(saveFileName + '.h5')
            self.target_model = keras.models.load_model(saveFileName + '.h5')
        else:
            self.model = create_model(self.learning_rate)
            self.target_model = create_model(self.learning_rate)
            # self.target_model.predict(np.zeros((1,state_size)))

    # n-step learning, get the truncated n-step return
    def get_n_step_info(self, n_step_buffer, gamma):
        """Return n step reward, next state, and done."""
        # info of the last transition
        reward, next_state, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            reward = r + gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return reward, next_state, done

    # these methods:sample,store are used in stratified experience replay
    def _make_pair(self, observation, action):
        return(hash(observation.tostring()),action)

    def _sample_index(self, nsteps=1):
        index_deque = self.pair_to_indices_dict.random_value()
        x = random.choice(index_deque)
        # Make sure the sampled index has room to bootstrap
        if (x - self.i) % self.size_now >= memory_size - nsteps:
            # It's too close to the pointer; recurse and try again
            return self._sample_index(nsteps)
        return x

    def sample(self, n):
        # Sample indices for the minibatch
        i = np.asarray([self._sample_index() for _ in range(n)])
        batches = [self.experience_replay[j] for j in i]
        return batches

    def store(self, experience):
        # n_step
        self.n_step_buffer.append(experience)
        if len(self.n_step_buffer) == self.n_step:
            reward, next_state, done = self.get_n_step_info(self.n_step_buffer, self.gamma)
            state, action = self.n_step_buffer[0][:2]
            # if memory is full
            if len(self.experience_replay) >= memory_size:
                s, a, r, ns, d = self.experience_replay[self.i]
                old_pair = self._make_pair(s, a)
                old_index_deque = self.pair_to_indices_dict[old_pair]
                old_index_deque.popleft()
                if not old_index_deque:
                    self.pair_to_indices_dict.pop(old_pair)
            new_pair = self._make_pair(state, action)
            if new_pair not in self.pair_to_indices_dict:
                self.pair_to_indices_dict[new_pair] = deque()
            self.pair_to_indices_dict[new_pair].append(self.i)
            if len(self.experience_replay) >= memory_size:
                self.experience_replay[self.i] = (state, action, reward, next_state, done)
            else:
                self.experience_replay.append((state, action, reward, next_state, done))
            self.i = (self.i + 1) % memory_size
            self.size_now = min(self.size_now+1, memory_size)


    def training(self):
        if len(self.experience_replay) >= self.replay_start_size:
            batches = self.sample(self.batch_size)
            states = np.vstack([data[0] for data in batches])
            actions = np.vstack([data[1] for data in batches])
            rewards = np.vstack([data[2] for data in batches])
            next_states = np.vstack([data[3] for data in batches])
            dones = np.vstack([data[4] for data in batches])
            self._train_body(states, actions, rewards, next_states, dones)

    @tf.function
    def _train_body(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            td_errors = self._compute_td_error_body(states, actions, rewards, next_states, dones)
            loss = tf.reduce_mean(tf.square(td_errors))  # huber loss seems no use
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return td_errors, loss

    @tf.function
    def _compute_td_error_body(self, states, actions, rewards, next_states, dones):
        rewards = tf.cast(tf.squeeze(rewards), dtype=tf.float32)
        dones = tf.cast(tf.squeeze(dones), dtype=tf.bool)
        actions = tf.cast(actions, dtype=tf.int32)  # (batch_size, 1)
        batch_size_range = tf.expand_dims(tf.range(self.batch_size), axis=1)  # (batch_size, 1)

        # get current q value
        current_q_indexes = tf.concat(values=(batch_size_range, actions), axis=1)  # (batch_size, 2)
        current_q = tf.gather_nd(self.model(states), current_q_indexes)  # (batch_size, )

        # get target q value using double dqn
        max_next_q_indexes = tf.argmax(self.model(next_states), axis=1, output_type=tf.int32)  # (batch_size, )
        indexes = tf.concat(values=(batch_size_range,
                                    tf.expand_dims(max_next_q_indexes, axis=1)), axis=1)  # (batch_size, 2)
        target_q = tf.gather_nd(self.target_model(next_states), indexes)  # (batch_size, )

        target_q = tf.where(dones, rewards, rewards + self.gamma * target_q)  # (batch_size, )
        # don't want change the weights of target network in backpropagation, so tf.stop_gradient()
        # but seems no use
        td_errors = tf.abs(current_q - tf.stop_gradient(target_q))
        return td_errors

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
for _ in range(num_trials):
    agent = DQNAgent()
    episode_rewards = []
    if agent.isTraining:
        scores_window = deque(maxlen=100)
        start = time.time()
        for episode in range(1, episodes + 1):
            rewards = 0
            state = env.reset()
            state = np.array([state])
            while True:
                action = agent.acting(state)
                next_state, reward, done, _ = env.step(action)
                rewards += reward
                next_state = next_state[None, :]
                reward = -10 if done else reward
                agent.store((state, action, reward, next_state, done))
                state = next_state
                agent.training()
                if done or rewards >= step_limit:
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
plt.savefig('QN_SER_nstep.png')
print('plotted')

import pickle
with open('qn_ser_nstep.pkl','wb') as f:
    pickle.dump(np.mean(plot_rewards, axis=0), f)
env.close()

