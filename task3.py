from dqn import DQN
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam


def build_model(input_shape, action_space):
    model = Sequential()
    model.add(Conv2D(
        filters=32,
        kernel_size=8,
        strides=(4, 4),
        padding='valid',
        activation='relu',
        input_shape=input_shape,
    ))
    model.add(Conv2D(
        filters=64,
        kernel_size=4,
        strides=(2, 2),
        padding='valid',
        activation='relu',
        input_shape=input_shape,
    ))
    model.add(Conv2D(
        filters=64,
        kernel_size=3,
        strides=(1, 1),
        padding='valid',
        activation='relu',
        input_shape=input_shape,
    ))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_space))
    model.compile(optimizer=Adam(), loss='mse')
    return model


def policy(state_c, action, reward, state_n, done, info):
    return reward + info['ale.lives'] * 0.2


if __name__ == '__main__':
    dqn = DQN('Breakout-v0', policy, build_model)
    dqn.train()
