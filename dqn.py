import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
import numpy as np
import numpy.random as npr
from tqdm import tqdm

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class DQN:
    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=np.product(self.env.observation_space.shape), activation='relu'))
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
        self.rewards = []

        self.batch_size = 32
        self.gamma = 0.99
        self.alpha = 1  # 0.1
        self.epsilon = np.linspace(1, 0.01, 2000)
        self.budget = 40000

    def train(self):
        with tqdm(total=self.budget) as progress:
            while True:

                # get initial state
                state_c = self.env.reset()
                self.env.render()
                done = False
                sum_rewards = 0
                while not done:

                    # select action with epsilon greedy
                    if npr.random() < self.epsilon[min(self.iterations, len(self.epsilon) - 1)]:
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(self.model.predict(state_c.reshape(1, -1))[0])

                    # perform action and store results in buffer
                    state_n, reward, done, _ = self.env.step(action)
                    sum_rewards += reward
                    if reward != 1:
                        reward = state_n[0] - 0.5

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

                # Add when done with game
                self.rewards.append(sum_rewards)