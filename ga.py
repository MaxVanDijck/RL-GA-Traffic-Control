import gym_cityflow
import gym
import random
import time

#Initialize Environment
env = gym.make('gym_cityflow:cityflow-v0', 
                configPath = 'data/4x4_config.json',
                episodeSteps = 500)

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
startTime = time.time()
popFitness = {}
for key, val in population.items():
    observation = env.reset()
    done = False
    count = 0
    cumulativeReward = 0
    totalCount = 0

    stepCounter = 0
    while done == False:
        totalCount += 1
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

    elapsedTime = time.time() - startTime
    startTime = time.time()
    popFitness[key] = cumulativeReward
    print(key + 'Finished, reward: ' + str(popFitness[key]) + ', Time Taken(s): ' + str(int(elapsedTime)))
