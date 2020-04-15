from dqn import DQN
import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def policy(state_c, action, reward, state_n, done):
    if reward == -1:
        return state_n[0] - 0.5
    else:
        return 1


if __name__ == '__main__':
    dqn = DQN('MountainCar-v0', policy)
    dqn.train()

    sns.lineplot(x=range(dqn.iteration), y=dqn.final_pos)
    plt.show()

    sns.lineplot(x=range(dqn.iteration), y=dqn.loss)
    plt.show()

    states = np.array(list(product(np.linspace(-1.2, 0.6, 100), np.linspace(-0.07, 0.07, 100))))
    actions = np.argmax(dqn.model1.predict(states), axis=1)

    df = pd.DataFrame({
        'Position': states[:, 0], 'Velocity': states[:, 1], 'Action': actions,
    })
    df['Action'] = df['Action'].astype('category')

    sns.scatterplot(data=df, x='Position', y='Velocity', hue='Action')
    plt.show()
