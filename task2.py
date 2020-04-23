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
        self.alpha = 0.99
        self.epsilon = np.linspace(1, 0.01, int(self.budget / 10))
        self.weight_update_frequency = int(self.budget / 100)
        self.frames_as_state = 1
        self.replay_size = int(self.budget / 10)

    @staticmethod
    def build_model(input_shape, action_space):
        model = Sequential()

        model.add(Dense(units=64, activation='relu', input_shape=input_shape))
        model.add(Dense(units=action_space, activation='linear'))

        model.compile(optimizer=Adam(), loss='mse')
        return model

    @staticmethod
    def policy(state_c, action, reward, state_n, done, info1, info2):
        info2['best'] = max(info2.get('best', -np.inf), state_c[-2])
        return max(0, state_n[-2] - info2['best'])

    @staticmethod
    def process_state(state):
        return state


if __name__ == '__main__':
    dqn = DQN('MountainCar-v0', 'task2', SettingsTask2())
    dqn.train(False)
    dqn.plots(sum)

    states = np.array(list(product(np.linspace(-1.2, 0.6, 100), np.linspace(-0.07, 0.07, 100))))
    actions = np.argmax(dqn.online.predict(states), axis=1)

    df = pd.DataFrame({
        'Position': states[:, 0], 'Velocity': states[:, 1], 'Action': actions,
    })
    df['Action'] = df['Action'].astype('category')

    sns.scatterplot(data=df, x='Position', y='Velocity', hue='Action')
    plt.savefig(join('output', dqn.name, 'policy.pdf'))
    plt.show()
