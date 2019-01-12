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

BeraterEnv.showStep = True
BeraterEnv.showDone = True

env = BeraterEnv.BeraterEnv( currentGraph )
print(env)
observation = env.reset()
print(observation)

for t in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()
print(observation)

######################################

import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
print(tf.__version__)

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

env = BeraterEnv.BeraterEnv( currentGraph )

wrapped_env = DummyVecEnv([lambda: BeraterEnv.BeraterEnv(currentGraph)])
monitored_env = VecMonitor(wrapped_env, log_dir)

# https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py
# https://github.com/openai/baselines/blob/master/baselines/common/models.py#L30
model = ppo2.learn(\
    env=monitored_env,\
    network='mlp',\
    # num_hidden=5000,\
    num_hidden=50,\
    # num_layers=3,\
    num_layers=2,\
    ent_coef=0.01,\
    # total_timesteps=500000)
    total_timesteps=5000)

# %time model = ppo2.learn(\
#     env=monitored_env,\
#     network='mlp',\
#     num_hidden=2000,\
#     num_layers=3,\
#     ent_coef=0.1,\
#     total_timesteps=500000)

# model = ppo2.learn(
#     env=monitored_env,\
#     layer_norm=True,\
#     network='mlp',\
#     num_hidden=2000,\
#     activation=tf.nn.relu,\
#     num_layers=3,\
#     ent_coef=0.03,\
#     total_timesteps=1000000)

# monitored_env = bench.Monitor(env, log_dir)
# https://en.wikipedia.org/wiki/Q-learning#Influence_of_variables
# %time model = deepq.learn(\
#         monitored_env,\
#         seed=42,\
#         network='mlp',\
#         lr=1e-3,\
#         gamma=0.99,\
#         total_timesteps=30000,\
#         buffer_size=50000,\
#         exploration_fraction=0.5,\
#         exploration_final_eps=0.02,\
#         print_freq=1000)

model.save('berater-ppo-v8.pkl')
monitored_env.close()

##################################################

from baselines.common import plot_util as pu
results = pu.load_results(log_dir)

import matplotlib.pyplot as plt
import numpy as np
r = results[0]
plt.ylim(0, .75)
# plt.plot(np.cumsum(r.monitor.l), r.monitor.r)
plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=100))

input("Press Enter to continue...")
