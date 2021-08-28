import gym_cityflow
import gym
import torch
import torch.nn as nn
import random
import json

#Initialize environment
configPath = 'data/1x1_config.json'
episodeSteps = 3600
env = gym.make('gym_cityflow:cityflow-v0')

#Traffic flow randomization
def randomiseFlow(configPath, episodeSteps):
    configDict = json.load(open(configPath))
    flowDict = json.load(open(configDict['dir'] + configDict['flowFile']))
    for i in range(len(flowDict)):
        randNum = random.randint(0, episodeSteps)
        flowDict[i]['startTime'] = randNum
        flowDict[i]['endTime'] = randNum
    json.dump(flowDict, open(configDict['dir'] + configDict['flowFile'], 'w'))

#Create DQN Model
class Model(nn.Module):
    def __init__(self, in_features, layer_size, out_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, out_features)
        self.relu - nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x