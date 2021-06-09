import numpy as np


class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_counter = 0

        self.states = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.next_states = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.rewards = np.zeros(self.mem_size, dtype=np.float32)
        self.actions = np.zeros(self.mem_size, dtype=np.int32)
        self.done = np.zeros(self.mem_size, dtype=np.bool)

    def store_step(self, state, action, reward, next_state, done):
        index = self.mem_counter % self.mem_size
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.done[index] = done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.states[batch]
        next_states = self.next_states[batch]
        rewards = self.rewards[batch]
        actions = self.actions[batch]
        done = self.done[batch]

        return states, actions, rewards, next_states, done
