import gym_cityflow
import gym

#Initialize Environment and check action space
env = gym.make('gym_cityflow:cityflow-v0', 
                configPath = '')

#initialise Population:
#parameters:
popSize = 100

population = []
for i in range(popSize):
    solution = []