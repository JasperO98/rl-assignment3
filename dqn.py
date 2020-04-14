import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
import numpy as np
import numpy.random as npr
from itertools import count

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class DQN:
    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.env.observation_space.shape[0], activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='softmax'))
        model.compile(optimizer=Adam(), loss='mse')
        return model

    def __init__(self, game):
        self.env = gym.make(game)
        self.model = self.build_model()
        self.replay = []

        self.batch_size = 256
        self.gamma = 0.99
        self.alpha = 0.2
        self.epsilon = np.linspace(1, 0.1, 1000)
        self.counter = count()

    def train(self):
        while True:
            state_c = self.env.reset()
            done = False

            while not done:
                if npr.random() < self.epsilon[min(next(self.counter), len(self.epsilon) - 1)]:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.model.predict(state_c.reshape(1, -1))[0])

                state_n, reward, done, _ = self.env.step(action)
                self.env.render()
                self.replay.append({'stateC': state_c, 'actionC': action, 'rewardN': reward, 'stateN': state_n})

            samples = npr.choice(self.replay, self.batch_size)
            samples_state_c = np.array([sample['stateC'] for sample in samples])
            samples_state_n = np.array([sample['stateN'] for sample in samples])
            samples_reward = np.array([sample['rewardN'] for sample in samples])
            samples_action = np.array([sample['actionC'] for sample in samples])

            target = self.model.predict(samples_state_c)
            target[range(len(target)), samples_action] *= 1 - self.alpha
            target[range(len(target)), samples_action] += self.alpha * (
                    samples_reward + self.gamma * np.max(self.model.predict(samples_state_n), axis=1)
            )

            self.model.fit(samples_state_c, target)


if __name__ == '__main__':
    dqn = DQN('MountainCar-v0')
    dqn.train()
