#This is a Neural Network model used to get Function approximization
#Reference Libraries

import torch
import torch.nn as nn
import torch.nn.functional as F

class QDeepNetwork(nn.Module):
    
    #setting defualt property while initializing
    #Number of states, actions to be set for the selected environment
    #set the random seed for repeated results
    #Neural network layers- Convolutional Layers 1 & 2 with 64 input features
    def __init__(self, state_size,action_size,seed, fc1_neurons=64,fc2_neurons=64):
        
        super(QDeepNetwork,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1  = nn.Linear(state_size,fc1_neurons)
        self.fc2  = nn.Linear(fc1_neurons, fc2_neurons)
        self.fc3  = nn.Linear(fc2_neurons,action_size)
        
    #Feed forward network
    def forward(self,state):
        x= F.relu(self.fc1(state))
        x= F.relu(self.fc2(x))
        return self.fc3(x)
