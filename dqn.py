import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
import numpy as np
import numpy.random as npr
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from itertools import product

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class DQN:
    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.env.observation_space.shape[0], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(optimizer=Adam(), loss='mse')
        return model

    def __init__(self, game):
        self.env = gym.make(game)
        self.model = self.build_model()
        self.replay = []
        self.loss = []
        self.iterations = 0

        self.batch_size = 32
        self.gamma = 0.99
        self.alpha = 0.1
        self.epsilon = np.linspace(1, 0.01, 2000)
        self.budget = 20000

    def train(self):
        with tqdm(total=self.budget) as progress:
            while True:

                # get initial state
                state_c = self.env.reset()
                self.env.render()
                done = False
                while not done:

                    # select action with epsilon greedy
                    if npr.random() < self.epsilon[min(self.iterations, len(self.epsilon) - 1)]:
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(self.model.predict(state_c.reshape(1, -1))[0])

                    # perform action and store results in buffer
                    state_n, reward, done, _ = self.env.step(action)
                    self.env.render()
                    self.replay.append({'stateC': state_c, 'actionC': action, 'rewardN': reward, 'stateN': state_n, 'doneN': done})
                    state_c = state_n

                    # select and process batch of samples
                    samples = npr.choice(self.replay, self.batch_size)
                    samples_state_c = np.array([sample['stateC'] for sample in samples])
                    samples_state_n = np.array([sample['stateN'] for sample in samples])
                    samples_reward = np.array([sample['rewardN'] for sample in samples])
                    samples_action = np.array([sample['actionC'] for sample in samples])
                    samples_done = np.array([sample['doneN'] for sample in samples])

                    # calculate and apply training targets for batch
                    target = self.model.predict(samples_state_c)
                    target[range(len(target)), samples_action] *= 1 - self.alpha
                    target[range(len(target)), samples_action] += self.alpha * (
                            samples_reward + self.gamma * np.max(self.model.predict(samples_state_n), axis=1) * ~samples_done
                    )
                    loss = self.model.fit(samples_state_c, target, epochs=1, verbose=0).history['loss'][0]

                    # end iteration
                    self.loss.append(loss)
                    self.iterations += 1
                    progress.update()
                    progress.desc = 'Loss ' + str(loss)

                    if self.iterations == self.budget:
                        return

    def plot_loss(self):
        sns.lineplot(x=range(self.iterations), y=self.loss)
        plt.show()

    def plot_choices_car(self):
        states = np.array(list(product(np.linspace(-1.2, 0.6, 100), np.linspace(-0.07, 0.07, 100))))
        actions = np.argmax(self.model.predict(states), axis=1)

        df = pd.DataFrame({
            'Position': states[:, 0], 'Velocity': states[:, 1], 'Action': actions,
        })
        df['Action'] = df['Action'].astype('category')

        sns.scatterplot(data=df, x='Position', y='Velocity', hue='Action')
        plt.show()

    def play_game(self):
        state = self.env.reset()
        self.env.render()
        done = False
        while not done:
            action = np.argmax(self.model.predict(state.reshape(1, -1))[0])
            state, _, done, _ = self.env.step(action)
            self.env.render()


if __name__ == '__main__':
    dqn = DQN('CartPole-v1')
    dqn.train()
    dqn.plot_loss()
    # dqn.plot_choices_car()
    dqn.play_game()
