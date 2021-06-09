import tensorflow as tf
from tensorflow import keras


class DuelingDeepQNetwork(keras.Model):
    def __init__(self, n_actions, observation_shape):
        super(DuelingDeepQNetwork, self).__init__()

        self.conv1 = keras.layers.Conv2D(16, 3, 3, input_shape=(*observation_shape,1))
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(128, activation='relu')
        self.dense2 = keras.layers.Dense(64, activation='relu')
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        expanded_state = tf.expand_dims(state, -1)
        x = self.conv1(expanded_state)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))
        return Q

    def advantage(self, state):
        expanded_state = tf.expand_dims(state, -1)
        x = self.conv1(expanded_state)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        A = self.A(x)
        return A
