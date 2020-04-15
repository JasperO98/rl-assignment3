from dqn import DQN, SettingsDQN
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import numpy as np
import cv2 as cv


class SettingsTask3(SettingsDQN):
    def __init__(self):
        self.budget = 20000
        self.batch_size = 32
        self.gamma = 0.99
        self.alpha = 1
        self.epsilon = np.linspace(1, 0.01, int(self.budget / 10))
        self.weight_update_frequency = int(self.budget / 100)
        self.frames_as_state = 4
        self.replay_size = int(self.budget / 10)

    @staticmethod
    def build_model(input_shape, action_space):
        model = Sequential()
        model.add(Conv2D(
            filters=16,
            kernel_size=8,
            strides=4,
            activation='relu',
            input_shape=input_shape,
        ))
        model.add(Conv2D(
            filters=32,
            kernel_size=4,
            strides=2,
            activation='relu',
            input_shape=input_shape,
        ))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(action_space, activation='linear'))
        model.compile(optimizer=Adam(), loss='mse')
        return model

    @staticmethod
    def policy(state_c, action, reward, state_n, done, info):
        return reward

    @staticmethod
    def process_state(state):
        state = cv.cvtColor(state, cv.COLOR_RGB2GRAY)
        state = state[32:-17, 8:-8]
        state = cv.resize(state, (64, 64), interpolation=cv.INTER_AREA)
        return state[:, :, np.newaxis]


if __name__ == '__main__':
    dqn = DQN('Breakout-v0', SettingsTask3())
    dqn.train(True)
