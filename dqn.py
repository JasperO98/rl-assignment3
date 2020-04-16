import gym
import numpy as np
import numpy.random as npr
from tqdm import tqdm
from collections import deque
import tensorflow as tf
from abc import ABC, abstractmethod
import json
from os.path import join

# limit GPU memory usage
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


class SettingsDQN(ABC):
    @staticmethod
    @abstractmethod
    def build_model(input_shape, action_space):
        pass

    @staticmethod
    @abstractmethod
    def policy(state_c, action, reward, state_n, done, info):
        pass

    @staticmethod
    @abstractmethod
    def process_state(state):
        pass


class DQN:
    def __init__(self, game, name, settings):
        self.name = name
        self.settings = settings
        self.env = gym.make(game)

        self.input_shape = list(self.settings.process_state(self.env.reset()).shape)
        self.channels = self.input_shape[-1]
        self.input_shape[-1] *= self.settings.frames_as_state

        self.model1 = settings.build_model(self.input_shape, self.env.action_space.n)
        self.model2 = settings.build_model(self.input_shape, self.env.action_space.n)
        self.update_target_model()

        self.replay = deque(maxlen=settings.replay_size)
        self.loss = []
        self.reward = []
        self.iteration = 0

    def update_target_model(self):
        self.model2.set_weights(self.model1.get_weights())

    def save(self):
        self.model1.save(join('output', self.name + '_' + str(self.iteration) + '.h5'))

        with open(join('output', self.name + '_loss.json'), 'w') as fp:
            fp.write(json.dumps(self.loss))
        with open(join('output', self.name + '_reward.json'), 'w') as fp:
            fp.write(json.dumps(self.reward))

    def train(self, render):
        with tqdm(total=self.settings.budget) as progress:
            while True:
                self.reward.append([])

                # get initial state
                state_c = self.env.reset()
                self.env.render() if render else None
                state_c = self.settings.process_state(state_c)
                state_c = np.tile(state_c, np.append(np.ones(len(self.input_shape) - 1, dtype=int), [self.settings.frames_as_state]))

                done = False
                while not done:
                    self.iteration += 1

                    # select action with epsilon greedy
                    if npr.random() < self.settings.epsilon[min(self.iteration, len(self.settings.epsilon)) - 1]:
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(self.model2.predict(np.expand_dims(state_c, axis=0))[0])

                    # perform action and store results in buffer
                    state_n, reward, done, info = self.env.step(action)
                    self.env.render() if render else None
                    state_n = self.settings.process_state(state_n)
                    state_n = np.append(state_c, state_n, axis=-1)[..., self.channels:]
                    self.reward[-1].append(reward)
                    reward = self.settings.policy(state_c, action, reward, state_n, done, info)
                    self.replay.append({'stateC': state_c, 'actionC': action, 'rewardN': reward, 'stateN': state_n, 'doneN': done})
                    state_c = state_n

                    # select and process batch of samples
                    samples = npr.choice(self.replay, self.settings.batch_size)
                    samples_state_c = np.array([sample['stateC'] for sample in samples])
                    samples_state_n = np.array([sample['stateN'] for sample in samples])
                    samples_reward = np.array([sample['rewardN'] for sample in samples])
                    samples_action = np.array([sample['actionC'] for sample in samples])
                    samples_done = np.array([sample['doneN'] for sample in samples])

                    # calculate and apply training targets for batch
                    target = self.model1.predict(samples_state_c)
                    target[range(len(target)), samples_action] *= 1 - self.settings.alpha
                    target[range(len(target)), samples_action] += self.settings.alpha * (
                            samples_reward + self.settings.gamma * np.max(self.model2.predict(samples_state_n), axis=1) * ~samples_done
                    )
                    loss = self.model1.fit(samples_state_c, target, epochs=1, verbose=0).history['loss'][0]
                    self.loss.append(loss)

                    # apply infrequent weight updates
                    if self.iteration % self.settings.weight_update_frequency == 0:
                        self.update_target_model()

                    # end iteration
                    progress.update()
                    progress.desc = 'Loss ' + str(loss)

                    if self.iteration == self.settings.budget:
                        self.reward.pop()
                        self.save()
                        return
                self.save()
