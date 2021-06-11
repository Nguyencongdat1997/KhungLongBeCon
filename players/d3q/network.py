import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


class DuelingDeepQNetwork(keras.Model):
    def __init__(self, n_actions, observation_shape):
        super(DuelingDeepQNetwork, self).__init__()

        self.reshape = keras.layers.Reshape((*observation_shape,1), input_shape=observation_shape)
        self.conv1 = keras.layers.Conv2D(64, 3, 3)        
        self.inorm1 = tfa.layers.InstanceNormalization(axis=3,center=True, scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
        self.conv2 = keras.layers.Conv2D(32, 3, 3)
        self.conv1 = keras.layers.Conv2D(16, 3, 3)        
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(512, activation='relu')
        self.dense2 = keras.layers.Dense(128, activation='relu')
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x= self.reshape(state)
        x = self.conv1(x)
        x = self.inorm1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))
        return Q

    def advantage(self, state):
        x= self.reshape(state)
        x = self.conv1(x)
        x = self.inorm1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        A = self.A(x)
        return A
