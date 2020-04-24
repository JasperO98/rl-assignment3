from keras.models import load_model
import gym
import numpy as np
import cv2 as cv
from task3 import SettingsTask3
from task2 import SettingsTask2
import time
from os.path import join


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    file = join('output','task2_RB5000_LR095_model4', 'model1_100000.h5')
    print(file)
    model = load_model(file)
    settings = SettingsTask2()
    input_shape = list(settings.process_state(env.reset()).shape)
    log_list = []
    for _ in range(100):
        done = False
        state = env.reset()
        # env.render()
        state = settings.process_state(state)
        state = np.tile(state, np.append(np.ones(len(input_shape) - 1, dtype=int), [settings.frames_as_state]))
        a = 0
        while not done:
            action = np.argmax(model.predict(np.expand_dims(state, axis=0))[0])
            state, reward, done, info = env.step(action)
            a += 1
            # env.render()
            state = settings.process_state(state)
            state = np.tile(state, np.append(np.ones(len(input_shape) - 1, dtype=int), [settings.frames_as_state]))
            time.sleep(0.05)
        print(a)
        print(_)
        log_list.append(a)
        env.close()
    print("---")
    print(log_list)
    print(np.mean(log_list))
    print(np.min(log_list))
    print(np.max(log_list))
    print(file)