from dqn import DQN
import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from keras.models import load_model
from gym import make
import itertools
from task2 import SettingsTask2
import os.path


class SettingsHere(SettingsTask2):
    def __init__(self, budget, gamma, alpha, weight_update_interval):
        self.budget = budget
        self.batch_size = 32
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = np.linspace(1, 0.01, int(self.budget / 10))
        self.weight_update_interval = weight_update_interval
        self.frames_as_state = 1
        self.replay_size = int(self.budget / 10)


def test_model(file, games_n, model_name_n):
    model = load_model(file)
    all_rewards = []
    env = make('MountainCar-v0')
    for _ in range(games_n):
        done = False
        reward_for_game = []
        state_c = env.reset()
        persistent = {}
        while not done:
            action = np.argmax(model.predict(np.expand_dims(state_c, axis=0))[0])
            state_n, reward, done, info = env.step(action)
            reward_for_game.append(SettingsHere.reward(state_c, action, reward, state_n, done, info, persistent))
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
    budget = [8192 << exponent for exponent in range(6)]  # [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    gamma = [0.99, 0.9]
    alpha = [1, 0.5]
    wuf = [64 << exponent for exponent in range(5)]  # [64, 128, 256, 512, 1024]
    wuf.append(1)
    return list(itertools.product(*[budget, gamma, alpha, wuf]))


if __name__ == '__main__':
    n_per_model = 4
    n_tests = 100
    settings = test_parameters()
    print(len(settings))
    for parameter in settings:
        model_name = '_'.join([str(x) for x in parameter])
        for i in range(n_per_model):
            model_name_n = model_name + '_V' + str(i + 1)
            if os.path.isdir(join('output', model_name_n)) == False:
                model_place = join('output', model_name_n, 'model1_' + str(parameter[0]) + '.h5')
                dqn = DQN('MountainCar-v0', model_name_n, SettingsHere(
                    parameter[0],
                    parameter[1],
                    parameter[2],
                    parameter[3],
                ))
                dqn.train(False)
                dqn.plots(sum)
                test_model(model_place, n_tests, model_name_n)
                plots(dqn)
            else:
                with open('log.txt', 'a') as myfile:
                    myfile.write(model_name_n + " already exists" + '\n')
