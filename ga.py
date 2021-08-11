import gym_cityflow
import gym
import random

#Initialize Environment
env = gym.make('gym_cityflow:cityflow-v0', 
                configPath = 'data/4x4_config.json',
                episodeSteps = 3600)

#initialise Population:
def createPopulation(popSize):
    population = {}
    for i in range(popSize):
        solution = []
        for j in range(12):
            singleActionStep = []
            for k in range(len(env.action_space.nvec)):
                singleActionStep.append(random.randint(0, env.action_space.nvec[k]-1))
            solution.append(singleActionStep)
        population['solution' + str(i)] = solution
    return population

population = createPopulation(popSize=100)

#Iterate through population and get fitness
popFitness = {}
for key, val in population.items():
    done = False
    count = 0
    cumulativeReward = 0

    stepCounter = 0
    while done == False:
        print(stepCounter)
        stepCounter +=1
        stepCounter = stepCounter % 10
        if stepCounter == 0:
            count += 1
            count = count % 11
        observation, reward, done, debug = env.step(population[key][count])
        for arr in reward:
            for i in range(len(arr)):
                if i != 0:
                    cumulativeReward += arr[i]

    popFitness[key] = cumulativeReward
    print(key + 'Finished, reward:' + str(reward))
    break