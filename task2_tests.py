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
from keras.models import load_model
from gym import make
import itertools


class SettingsTask2(SettingsDQN):
    def __init__(self, budget=20000, gamma=0.99, alpha=0.99, weight_update_frequency=128):
        self.budget = budget
        self.batch_size = 32
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = np.linspace(1, 0.01, int(self.budget / 10))
        self.weight_update_frequency = weight_update_frequency
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
    def reward(state_c, action, reward, state_n, done, info1, info2):
        info2['best'] = max(info2.get('best', -np.inf), state_c[-2])
        return max(0, state_n[-2] - info2['best'])

    @staticmethod
    def process_state(state):
        return state


def test_model(file, games_n, model_name_n):
    model = load_model(file)
    all_rewards = []

    for _ in range(games_n):
        done = False
        reward_for_game = []
        state_c = env.reset()
        persistent = {}
        while not done:
            action = np.argmax(model.predict(np.expand_dims(state_c, axis=0))[0])
            state_n, reward, done, info = env.step(action)
            reward_for_game.append(SettingsTask2.reward(state_c, action, reward, state_n, done, info, persistent))
            state_c = state_n
        all_rewards.append(sum(reward_for_game))

    with open('stats_model.txt', 'a') as myfile:
        myfile.write('AVG\t' + model_name_n + '\t' + str(np.mean(all_rewards)) + '\n')
        myfile.write('MAX\t' + model_name_n + '\t' + str(np.max(all_rewards)) + '\n')
        myfile.write('MIN\t' + model_name_n + '\t' + str(np.min(all_rewards)) + '\n')


def plots(dqn):
    states = np.array(list(product(np.linspace(-1.2, 0.6, 100), np.linspace(-0.07, 0.07, 100))))
    actions = np.argmax(dqn.online.predict(states), axis=1)

    df = pd.DataFrame({
        'Position': states[:, 0], 'Velocity': states[:, 1], 'Action': actions,
    })
    df['Action'] = df['Action'].astype('category')

    sns.scatterplot(data=df, x='Position', y='Velocity', hue='Action')
    plt.tight_layout()
    plt.savefig(join('output', dqn.name, 'policy.pdf'))
    plt.close()


def test_parameters():
    budget = [8192 << exponent for exponent in range(8)]  # [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    gamma = [0.99, 0.9]
    alpha = [1, 0.5]
    wuf = [64 << exponent for exponent in range(5)]  # [64, 128, 256, 512, 1024]
    wuf.append(1)
    return list(itertools.product(*[budget, gamma, alpha, wuf]))


if __name__ == '__main__':
    env = make('MountainCar-v0')
    n_per_model = 4
    n_tests = 100
    settings = test_parameters()
    print(len(settings))
    for parameter in settings:
        model_name = '_'.join([str(x) for x in parameter])
        print(model_name)
        for i in range(n_per_model):
            model_name_n = model_name + '_V' + str(i + 1)
            model_place = join('output', model_name_n, 'model1_' + str(parameter[0]) + '.h5')
            dqn = DQN('MountainCar-v0', model_name_n, SettingsTask2(parameter[0],
                                                                    parameter[1],
                                                                    parameter[2],
                                                                    parameter[3]))
            dqn.train(False)
            dqn.plots(sum)
            test_model(model_place, n_tests, model_name_n)
            plots(dqn)
