from keras.models import load_model
import gym
import numpy as np
import cv2 as cv
from task3 import SettingsTask3
import time
from os.path import join


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    model = load_model(join('output', 'task3.h5'))
    settings = SettingsTask3()
    input_shape = list(settings.process_state(env.reset()).shape)

    done = False
    state = env.reset()
    
    env.render()
    
    state = settings.process_state(state)
    state = np.tile(state, np.append(np.ones(len(input_shape) - 1, dtype=int), [settings.frames_as_state]))

    while not done:
        action = np.argmax(model.predict(np.expand_dims(state, axis=0))[0])
        state, reward, done, info = env.step(action)
        
        env.render()

        state = settings.process_state(state)
        state = np.tile(state, np.append(np.ones(len(input_shape) - 1, dtype=int), [settings.frames_as_state]))

        time.sleep(0.05)

    env.close()
    
