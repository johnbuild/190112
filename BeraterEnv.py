import numpy as np
import random

import gym
from gym.utils import seeding
from gym import spaces

import GraphDefinitions

class BeraterEnv(gym.Env):
    """
    The Berater Problem

    Actions: 
    There are 4 discrete deterministic actions, each choosing one direction
    """
    metadata = {'render.modes': ['ansi']}
    
    showStep = False
    showDone = True
    envEpisodeModulo = 100
    number_of_consultants = 2

    def __init__(self, graph):
        self.map = graph
        max_paths = 4
        self.action_space = spaces.Discrete(max_paths)
      
        positions = len(self.map)
        # observations: position, reward of all 4 local paths, rest reward of all locations
        # non existing path is -1000 and no position change
        # look at what #getObservation returns if you are confused
        low = np.append(np.append([0], np.full(max_paths, -1000)), np.full(positions, 0))
        high = np.append(np.append([positions - 1], np.full(max_paths, 1000)), np.full(positions, 1000))
        self.observation_space = spaces.Box(low=low,
                                             high=high,
                                             dtype=np.float32)
        self.reward_range = (-1, 1)

        self.totalReward = 0
        self.stepCount = 0
        self.isDone = False

        self.envReward = 0
        self.envEpisodeCount = 0
        self.envStepCount = 0

        self.reset()
        self.optimum = self.calculate_customers_reward()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def iterate_path(self, state, action):
        paths = self.map[state]
        if action < len(paths):
          return paths[action]
        else:
          # sorry, no such action, stay where you are and pay a high penalty
          return (state, 1000)
      
    def step(self, action):
        destination, cost = self.iterate_path(self.state, action)
        lastState = self.state
        customerReward = self.customer_reward[destination]
        reward = (customerReward - cost) / self.optimum
        
        pureReward = 0
        if (cost != 1000 and cost != 1) pureReward = reward
        self.totalPureReward += pureReward

        self.state = destination
        self.customer_visited(destination)
        done = destination == 'S' and self.all_customers_visited()

        stateAsInt = GraphDefinitions.state_name_to_int(self.state)
        self.totalReward += reward

        self.stepCount += 1
        self.envReward += reward
        self.envStepCount += 1

        if self.showStep:
            print( "Episode: " + ("%4.0f  " % self.envEpisodeCount) + 
                   " Step: " + ("%4.0f  " % self.stepCount) + 
                   lastState + ' --' + str(action) + '-> ' + self.state + 
                   ' R=' + ("% 2.2f" % reward) + ' totalR=' + ("% 3.2f" % self.totalReward) + 
                   ' cost=' + ("%4.0f" % cost) + ' customerR=' + ("%4.0f" % customerReward) + ' optimum=' + ("%4.0f" % self.optimum)      
                   )

        if done and not self.isDone:
            self.envEpisodeCount += 1
            if BeraterEnv.showDone:
                episodes = BeraterEnv.envEpisodeModulo
                if (self.envEpisodeCount % BeraterEnv.envEpisodeModulo) == 0:
                    print( "Done: " + 
                        ("episodes=%6.0f  " % self.envEpisodeCount) + 
                        ("avgSteps=%6.2f  " % (self.envStepCount/episodes)) + 
                        ("avgTotalReward=% 3.2f" % (self.envReward/episodes) )
                        )
                    self.envReward = 0
                    self.envStepCount = 0

        self.isDone = done
        observation = self.getObservation(stateAsInt)
        info = {"from": self.state, "to": destination}

        return observation, reward, done, info

    def getObservation(self, position):
        result = np.array([ position, 
                               self.getPathObservation(position, 0),
                               self.getPathObservation(position, 1),
                               self.getPathObservation(position, 2),
                               self.getPathObservation(position, 3)
                              ],
                             dtype=np.float32)
        all_rest_rewards = list(self.customer_reward.values())
        result = np.append(result, all_rest_rewards)
        return result

    def getPathObservation(self, position, path):
        source = GraphDefinitions.int_to_state_name(position)
        paths = self.map[source]
        if path < len(paths):
          target, cost = paths[path]
          reward = self.customer_reward[target] 
          result = reward - cost
        else:
          result = -1000

        return result

    def customer_visited(self, customer):
        self.customer_reward[customer] = 0

    def all_customers_visited(self):
        sum = 0
        for value in self.customer_reward.values():
            sum += value
        return sum == 0

    def calculate_customers_reward(self):
        result = GraphDefinitions.getTotalReward(self.map,self.customer_reward)
        return result

      
    def modulate_reward(self):
        number_of_customers = len(self.map) - 1
        number_per_consultant = int(number_of_customers/BeraterEnv.number_of_consultants)
    #       number_per_consultant = int(number_of_customers/1.5)
        self.customer_reward = {
            'S': 0
        }
        if (BeraterEnv.number_of_consultants > 1):
            for customer_nr in range(1, number_of_customers + 1):
                self.customer_reward[GraphDefinitions.int_to_state_name(customer_nr)] = 0
            
            # every consultant only visits a few random customers
            samples = random.sample(range(1, number_of_customers + 1), k=number_per_consultant)
            key_list = list(self.customer_reward.keys())
            for sample in samples:
                self.customer_reward[key_list[sample]] = 1000
        else:
            for customer_nr in range(1, number_of_customers + 1):
                self.customer_reward[GraphDefinitions.int_to_state_name(customer_nr)] = 1000

      
    def reset(self):
        self.totalPureReward = 0
        self.totalReward = 0
        self.stepCount = 0
        self.isDone = False

        self.modulate_reward()
        self.state = 'S'
        return self.getObservation(GraphDefinitions.state_name_to_int(self.state))
      
    def render(self):
        print("graph definition  : ", end='')
        print(self.map)
        print("customer_reward   : " ,end='')
        print(self.customer_reward)
        print("optimum           : ", end='')
        print(self.optimum)
        print("observation       : ", end='')
        stateAsInt = GraphDefinitions.state_name_to_int(self.state)
        print(self.getObservation(stateAsInt))