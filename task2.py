from dqn import DQN, SettingsDQN
import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from os.path import join


class SettingsTask2(SettingsDQN):
    def __init__(self):
        self.budget = 20000
        self.batch_size = 32
        self.gamma = 0.99
        self.alpha = 1
        self.epsilon = np.linspace(1, 0.01, int(self.budget / 10))
        self.weight_update_frequency = int(self.budget / 100)
        self.frames_as_state = 1
        self.replay_size = int(self.budget / 10)

    @staticmethod
    def build_model(input_shape, action_space):
        model = Sequential()
        model.add(Dense(16, input_shape=input_shape, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(action_space, activation='linear'))
        model.compile(optimizer=Adam(), loss='mse')
        return model

    @staticmethod
    def policy(state_c, action, reward, state_n, done, info):
        return reward

    @staticmethod
    def process_state(state):
        return state


if __name__ == '__main__':
    dqn = DQN('MountainCar-v0', 'task2', SettingsTask2())
    dqn.train(True)

    sns.lineplot(x=range(len(dqn.reward)), y=[sum(x) for x in dqn.reward])
    plt.savefig(join('output', 'task2_reward.pdf'))
    plt.show()

    sns.lineplot(x=range(len(dqn.loss)), y=dqn.loss)
    plt.savefig(join('output', 'task2_loss.pdf'))
    plt.show()

    states = np.array(list(product(np.linspace(-1.2, 0.6, 100), np.linspace(-0.07, 0.07, 100))))
    actions = np.argmax(dqn.model1.predict(states), axis=1)

    df = pd.DataFrame({
        'Position': states[:, 0], 'Velocity': states[:, 1], 'Action': actions,
    })
    df['Action'] = df['Action'].astype('category')

    sns.scatterplot(data=df, x='Position', y='Velocity', hue='Action')
    plt.savefig(join('output', 'task2_policy.pdf'))
    plt.show()
