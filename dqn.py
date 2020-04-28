import gym
import numpy as np
import numpy.random as npr
from tqdm import tqdm
import tensorflow as tf
from abc import ABC, abstractmethod
import json
from os.path import join
from glob import glob
from natsort import natsorted
from keras.models import load_model
from os import makedirs, environ
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import count
from random import seed
from scipy.special import softmax

# limit GPU memory usage
for device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)

# set random generator seeds
SEED = 678934502
environ['PYTHONHASHSEED'] = str(SEED)
seed(SEED)
npr.seed(SEED)
tf.random.set_seed(SEED)


class SettingsDQN(ABC):
    @staticmethod
    @abstractmethod
    def build_model(input_shape, action_space):
        pass

    @staticmethod
    @abstractmethod
    def reward(state_c, action, reward, state_n, done, info1, info2):
        pass

    @staticmethod
    @abstractmethod
    def process_state(state):
        pass


class ReplayBuffer:
    def __init__(self, maxlen, input_shape):
        input_shape = [maxlen] + input_shape

        self.state_c = np.zeros(input_shape, dtype=np.float32)
        self.action_c = np.zeros(maxlen, dtype=np.uint8)
        self.reward_n = np.zeros(maxlen, dtype=np.float32)
        self.state_n = np.zeros(input_shape, dtype=np.float32)
        self.done_n = np.zeros(maxlen, dtype=np.bool)

        self.maxlen = maxlen
        self.appends = 0

    def append(self, state_c, action_c, reward_n, state_n, done_n):
        self.state_c[self.appends % self.maxlen] = state_c
        self.action_c[self.appends % self.maxlen] = action_c
        self.reward_n[self.appends % self.maxlen] = reward_n
        self.state_n[self.appends % self.maxlen] = state_n
        self.done_n[self.appends % self.maxlen] = done_n

        self.appends += 1

    def sample(self, size):
        length = min(self.appends, self.maxlen)
        size = min(size, length)
        indices = npr.choice(length, size, False, softmax(self.reward_n[:length]))
        return self.state_c[indices], self.action_c[indices], self.reward_n[indices], self.state_n[indices], self.done_n[indices]

    def save(self, name):
        np.savez(
            join('output', name, 'replay.npz'),
            state_c=self.state_c,
            action_c=self.action_c,
            reward_n=self.reward_n,
            state_n=self.state_n,
            done_n=self.done_n,
            maxlen=self.maxlen,
            appends=self.appends,
        )

    @staticmethod
    def load(name):
        replay = ReplayBuffer(0, [0])
        file = np.load(join('output', name, 'replay.npz'))

        replay.state_c = file['state_c']
        replay.action_c = file['action_c']
        replay.reward_n = file['reward_n']
        replay.state_n = file['state_n']
        replay.done_n = file['done_n']
        replay.maxlen = file['maxlen']
        replay.appends = file['appends']

        return replay


class DQN:
    def __init__(self, game, name, settings):
        self.name = name
        self.settings = settings
        self.env = gym.make(game)
        self.env.seed(SEED)

        self.input_shape = list(self.settings.process_state(self.env.reset()).shape)
        self.channels = self.input_shape[-1]
        self.input_shape[-1] *= self.settings.frames_as_state
        self.action_space = self.env.action_space.n

        models = natsorted(glob(join('output', name, 'model1_*.h5')))

        if len(models) > 0:
            self.online = load_model(models[-1])
            self.target = load_model(join('output', name, 'model2.h5'))
            self.replay = ReplayBuffer.load(self.name)

            with open(join('output', name, 'loss.json'), 'r') as fp:
                self.loss = json.load(fp)
            with open(join('output', name, 'reward.json'), 'r') as fp:
                self.reward = json.load(fp)
            self.iteration = int(models[-1].split('_')[-1].split('.')[-2])

        else:
            self.online = settings.build_model(self.input_shape, self.action_space)
            self.target = settings.build_model(self.input_shape, self.action_space)
            self.update_target_model()

            self.replay = ReplayBuffer(settings.replay_size, self.input_shape)
            self.loss = []
            self.reward = []
            self.iteration = 0

    def update_target_model(self):
        self.target.set_weights(self.online.get_weights())

    def save(self):
        makedirs(join('output', self.name), exist_ok=True)

        self.online.save(join('output', self.name, 'model1_' + str(self.iteration) + '.h5'))
        self.target.save(join('output', self.name, 'model2.h5'))
        self.replay.save(self.name)

        with open(join('output', self.name, 'loss.json'), 'w') as fp:
            json.dump(self.loss, fp)
        with open(join('output', self.name, 'reward.json'), 'w') as fp:
            json.dump(self.reward, fp)

    def plots(self, reduce):
        makedirs(join('output', self.name), exist_ok=True)

        ax = sns.lineplot(x=range(len(self.reward)), y=[reduce(x) for x in self.reward])
        ax.set(ylabel='Cumulative Reward', xlabel='Game #')
        plt.tight_layout()
        plt.savefig(join('output', self.name, 'reward.pdf'))
        plt.close()

        ax = sns.lineplot(x=range(len(self.loss)), y=self.loss)
        ax.set(yscale='log', ylabel='Training Loss', xlabel='Iteration #')
        plt.tight_layout()
        plt.savefig(join('output', self.name, 'loss.pdf'))
        plt.close()

    def train(self, render):
        with tqdm(total=self.settings.budget, desc='Waiting ...', initial=self.iteration) as progress:
            if self.iteration >= self.settings.budget:
                return
            for i in count(1):
                self.reward.append([])
                persistent = {}

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
                        action = npr.randint(self.action_space)
                    else:
                        action = np.argmax(self.online.predict(np.expand_dims(state_c, 0))[0])

                    # perform action and store results in buffer
                    state_n, reward, done, info = self.env.step(action)
                    self.env.render() if render else None
                    state_n = self.settings.process_state(state_n)
                    state_n = np.append(state_c, state_n, axis=-1)[..., self.channels:]
                    reward = self.settings.reward(state_c, action, reward, state_n, done, info, persistent)
                    self.reward[-1].append(reward)
                    self.replay.append(state_c, action, reward, state_n, done)
                    state_c = state_n

                    # select and process batch of samples
                    samples_state_c, samples_action, samples_reward, samples_state_n, samples_done = self.replay.sample(self.settings.batch_size)

                    # calculate and apply training targets for batch
                    target = self.online.predict(samples_state_c)
                    target[range(len(target)), samples_action] *= 1 - self.settings.alpha
                    target[range(len(target)), samples_action] += self.settings.alpha * (
                            samples_reward + self.settings.gamma * np.max(self.target.predict(samples_state_n), axis=1) * ~samples_done
                    )
                    loss = self.online.fit(
                        x=samples_state_c, y=target, batch_size=self.settings.batch_size, epochs=1, verbose=0,
                    ).history['loss'][0]
                    self.loss.append(loss)

                    # apply infrequent weight updates
                    if self.iteration % self.settings.weight_update_interval == 0:
                        self.update_target_model()

                    # end iteration
                    progress.update()
                    if len(self.reward) >= 2:
                        progress.desc = 'Prev. Cum. Reward = ' + str(round(sum(self.reward[-2]), 4)) + ', Loss = ' + str(round(loss, 4))

                    if self.iteration == self.settings.budget:
                        self.reward.pop()
                        self.save()
                        return
                if i % 100 == 0:
                    self.save()
