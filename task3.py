from dqn import DQN, SettingsDQN
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import numpy as np


class SettingsTask3(SettingsDQN):
    def __init__(self):
        self.budget = 20000
        self.batch_size = 32
        self.gamma = 0.99
        self.alpha = 1
        self.epsilon = np.linspace(1, 0.01, int(self.budget / 10))
        self.weight_update_frequency = int(self.budget / 100)
        self.frames_as_state = 4

    @staticmethod
    def build_model(input_shape, action_space):
        model = Sequential()
        model.add(Conv2D(
            filters=32,
            kernel_size=8,
            strides=(4, 4),
            padding='valid',
            activation='relu',
            input_shape=input_shape,
        ))
        model.add(Conv2D(
            filters=64,
            kernel_size=4,
            strides=(2, 2),
            padding='valid',
            activation='relu',
            input_shape=input_shape,
        ))
        model.add(Conv2D(
            filters=64,
            kernel_size=3,
            strides=(1, 1),
            padding='valid',
            activation='relu',
            input_shape=input_shape,
        ))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(action_space, activation='linear'))
        model.compile(optimizer=Adam(), loss='mse')
        return model

    @staticmethod
    def policy(state_c, action, reward, state_n, done, info):
        return reward + info['ale.lives']


if __name__ == '__main__':
    dqn = DQN('Breakout-v0', SettingsTask3())
    dqn.train()
