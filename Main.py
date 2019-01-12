import numpy as np
import random

import gym
from gym.utils import seeding
from gym import spaces

import BeraterEnv
import GraphDefinitions  

#########################################

log_dir = '/temp/_berater/'
currentGraph = GraphDefinitions.getSmallGraph()

env = BeraterEnv.BeraterEnv( currentGraph )
print(env.reset())
print(env.customer_reward)

#########################################

print("--- Test environment ---")

BeraterEnv.BeraterEnv.showStep = True
BeraterEnv.BeraterEnv.showDone = True

env = BeraterEnv.BeraterEnv( currentGraph )
print("Environment        : ", end='')
print(env)
observation = env.reset()
print("Initial Observation: ", end='')
print(observation)

for t in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()
print("Last Observation   : ", end='')
print(observation)

######################################

import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
print("Tensorflow version : " + str(tf.__version__) )

########################################

# https://github.com/openai/baselines/blob/master/baselines/deepq/experiments/train_pong.py
# log_dir = logger.get_dir()


import gym
from baselines import bench
from baselines import logger

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_monitor import VecMonitor
from baselines.ppo2 import ppo2

BeraterEnv.BeraterEnv.showStep = False
BeraterEnv.BeraterEnv.showDone = True

print("--- PPO2 learn ---")

env = BeraterEnv.BeraterEnv( currentGraph )

wrapped_env = DummyVecEnv([lambda: BeraterEnv.BeraterEnv(currentGraph)])
monitored_env = VecMonitor(wrapped_env, log_dir)

# https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py
# https://github.com/openai/baselines/blob/master/baselines/common/models.py#L30
model = ppo2.learn(\
    env=monitored_env,\
    network='mlp',\
    num_hidden=50,\
    num_layers=2,\
    ent_coef=0.01,\
    total_timesteps=500)

model.save('berater-ppo-v8.pkl')
monitored_env.close()

##################################################

print("--- Plot ---")

from baselines.common import plot_util as pu
results = pu.load_results(log_dir)

import matplotlib.pyplot as plt
import numpy as np
r = results[0]
plt.ylim(0, .75)
# plt.plot(np.cumsum(r.monitor.l), r.monitor.r)
plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=100))

##################################################

import numpy as np 

print("--- Enjoy ---")

observation = env.reset()
env.render()
state = np.zeros((1, 2*128))
dones = np.zeros((1))

BeraterEnv.BeraterEnv.showStep = True
BeraterEnv.BeraterEnv.showDone = True
BeraterEnv.BeraterEnv.number_of_consultants = 1

for t in range(1000):
    actions, _, state, _ = model.step(observation, S=state, M=dones)
    observation, reward, done, info = env.step(actions[0])
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()

input("Press Enter to continue...")
