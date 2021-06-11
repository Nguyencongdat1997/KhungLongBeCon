import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


from players.d3q.network import DuelingDeepQNetwork
from utils.replay_buffer import ReplayBuffer


class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, coldstart, epsilon_dec=1e-3, epsilon_end=0.01,
                 mem_size=100, replace=256):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.replace = replace
        self.batch_size = batch_size
        self.coldstart = coldstart
        self.mem_size = mem_size

        self.learned_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_active = DuelingDeepQNetwork(n_actions, input_dims)
        self.q_frozen = DuelingDeepQNetwork(n_actions, input_dims)

        self.q_active.build(input_shape =(batch_size,*input_dims))
        self.q_frozen.build(input_shape =(batch_size,*input_dims))
        self.q_active.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
        self.q_frozen.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    def store_step(self, state, action, reward, next_state, done):
        self.memory.store_step(state, action, reward, next_state, done)

    def play(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            actions = self.q_active.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        return action

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        if self.learned_step_counter % self.replace == 0:
            self.q_frozen.set_weights(self.q_active.get_weights())

        # get data
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        q_pred = self.q_active(states)
        #print(q_pred)
        q_next = self.q_frozen(next_states)
        q_target = q_pred.numpy()
        max_next_actions = tf.math.argmax(self.q_active(next_states), axis=1)
        #print(max_next_actions)
        for i, terminated in enumerate(dones):
            q_target[i, actions[i]] = rewards[i] + self.gamma * q_next[i, int(max_next_actions[i])] * (1 - int(dones[i]))

        # train
        self.q_active.train_on_batch(states, q_target)

        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)
        self.learned_step_counter += 1

    def train(self, env, n_games):
        if self.coldstart > 0:
            print('Cold start ----------')       
            steps = 0        
            while steps < self.coldstart:
                done = False
                observation = env.reset()
                while not done:
                    steps += 1
                    action = np.random.choice(self.action_space)
                    reward, next_observation, done = env.step(action)
                    self.store_step(observation, action, reward, next_observation, done)
                    observation = next_observation
                print(' Steps', steps)

        print('Start training ----------')       
        scores = []
        eps_history = []
        steps = 0
        for i in range(n_games):
            done = False
            score = 0
            observation = env.reset()
            while not done:
                steps += 1
                action = self.play(observation)
                reward, next_observation, done = env.step(action)
                score += reward
                self.store_step(observation, action, reward, next_observation, done)
                observation = next_observation
            eps_history.append(self.epsilon)
            scores.append(score)
            avg_score = np.mean(scores[-5:])
            print(' Episode', i, '- Trained steps', steps, '- Score %.1f' % score, '- Avg_score %.1f ' % avg_score, 
                    '- Epsilon %.001f ' % self.epsilon, '- Learned {} steps'.format(self.learned_step_counter))

            for i in range(int(self.mem_size/self.batch_size)):
                self.learn()  # TODO: decide to learn in the end of the episode or in each steps
        print('End training ----------')

    def save_model(self, train_dir):
        file_name = train_dir + '/d3qn_' + str(self.learned_step_counter) + '/model'
        self.q_active.save_weights(file_name, save_format='tf')

    def load_model(self, train_dir, learned_steps=100):
        file_name = train_dir + '/d3qn_' + str(learned_steps) + '/model'
        self.q_active.load_weights(file_name)
        self.q_frozen.set_weights(self.q_active.get_weights())
