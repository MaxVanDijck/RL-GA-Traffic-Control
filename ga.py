import gym_cityflow
import gym
import random
import time
import json
import operator

#Initialize Environment
configPath = 'data/1x1_config.json'
episodeSteps = 3600
env = gym.make('gym_cityflow:cityflow-v0', 
                configPath = configPath,
                episodeSteps = episodeSteps)

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

def randomiseFlow(configPath, episodeSteps):
    configDict = json.load(open(configPath))
    flowDict = json.load(open(configDict['dir'] + configDict['flowFile']))
    for i in range(len(flowDict)):
        randNum = random.randint(0, episodeSteps)
        flowDict[i]['startTime'] = randNum
        flowDict[i]['endTime'] = randNum
    json.dump(flowDict, open(configDict['dir'] + configDict['flowFile'], 'w'))


#Iterate through population and get fitness
startTime = time.time()
popFitness = {}

numTests = 20

def score(population, nonExp=False):
    startTime = time.time()
    popFitness = {}
    numTests = 20
    for key, val in population.items():
        avgReward = 0
        for i in range(numTests):
            #randomize flow file
            randomiseFlow(configPath=configPath, episodeSteps=episodeSteps)

            env = gym.make('gym_cityflow:cityflow-v0', 
                    configPath = configPath,
                    episodeSteps = episodeSteps)
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
                if nonExp:
                    reward = env.getReward_nonExp()
                for arr in reward:
                    for j in range(len(arr)):
                        if j != 0:
                            cumulativeReward += arr[j]
            avgReward += cumulativeReward
        avgReward = avgReward / numTests

        elapsedTime = time.time() - startTime
        startTime = time.time()
        popFitness[key] = avgReward
        print(key + 'Finished, reward: ' + str(popFitness[key]) + ', Time Taken(s): ' + str(int(elapsedTime)))

    sortedFitness = sorted(popFitness.items(), key=operator.itemgetter(1))
    return sortedFitness

def crossover(rewards, population):
    for i in range(len(rewards)):
        if i % 2 == 0:
            solution1 = population[rewards[i][0]]
            solution2 = population[rewards[i+1][0]]

            crossoverPoint1 = random.randint(0, len(solution1))

            while True:
                crossoverPoint2 = random.randint(0, len(solution2))
                if crossoverPoint2 != crossoverPoint1:
                    break

            if crossoverPoint1 > crossoverPoint2:
                temp = crossoverPoint1
                crossoverPoint1 = crossoverPoint2
                crossoverPoint2 = temp

            tempSolution1 = []
            tempSolution2 = []
            
            for j in range(len(solution1)):
                if j < crossoverPoint1:
                    tempSolution1.append(solution1[j])
                    tempSolution2.append(solution2[j])
                elif j >= crossoverPoint1 and j <= crossoverPoint2:
                    tempSolution1.append(solution2[j])
                    tempSolution2.append(solution1[j])
                else:
                    tempSolution1.append(solution1[j])
                    tempSolution2.append(solution2[j])

            population[rewards[i][0]] = tempSolution1
            population[rewards[i+1][0]] = tempSolution2
            
    return population

def mutate(rewards, population, mutations, mutateRatio):
    for i in range(int(len(rewards) * mutateRatio)):
        solution = population[rewards[i][0]]
        for j in range(mutations):
            position = random.randint(0, len(solution)-1)
            solution[position] = [random.randint(0, env.action_space.nvec[0]-1)]

        population[rewards[i][0]] = solution

    return population

def savePop(population, path):
    json.dump(population, open(path, 'w'))

def openPop():
    return json.load(open('population.json'))


populationExp = createPopulation(popSize=100)
populationNonExp = createPopulation(popSize=100)


for i in range(100):
    print('Scoring Populations')
    print('Exp Scoring')
    sortedFitnessExp = score(populationExp)
    print('Non-Exp Scoring')
    sortedFitnessNonExp = score(populationNonExp, nonExp=True)
    print('Starting Comparison Run (Exp Reward Population evaluated with Non-Exp reward)')
    comparisonFitness = score(populationExp, nonExp=True)
    print('Exponential Reward Sorted Fitness')
    print(sortedFitnessExp)
    print('Non Exponential Reward Sorted Fitness')
    print(sortedFitnessNonExp)
    print('Saving Populations')
    savePop(populationExp, 'popExp{}.json'.format(i))
    savePop(populationNonExp, 'popNonExp{}.json'.format(i))

    print('Population Crossover')
    populationExp = crossover(rewards=sortedFitnessExp, population=populationExp)
    populationNonExp = crossover(rewards=sortedFitnessNonExp, population=populationNonExp)

    print('Mutating Population')
    populationExp = mutate(rewards=sortedFitnessExp, population=populationExp, mutations=1, mutateRatio=0.5)
    populationNonExp = mutate(rewards=sortedFitnessNonExp, population=populationNonExp, mutations=1, mutateRatio=0.5)