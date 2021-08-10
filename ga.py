import gym_cityflow
import gym
import random

#Initialize Environment
env = gym.make('gym_cityflow:cityflow-v0', 
                configPath = 'data/1x1_config.json',
                episodeSteps = 3600)

#initialise Population:
#parameters:
popSize = 100
population = {}
for i in range(popSize):
    solution = []
    for j in range(12):
        solution.append(random.randint(0, 8))

    population['solution' + str(i)] = solution