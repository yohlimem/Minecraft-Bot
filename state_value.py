from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch import optim

class StateValue(nn.Module):
    def __init__(self, state_size:int, hidden_amount:int =3, layer_size:int =256):
        super().__init__()

        '''
        state_size size is a number bigger than 1 that represents the amount of inputs for the network
        hidden_amount is a number that represents the amount of hidden layers
        layer_size is a number bigger than 1 that represents the amount of neurons in a hidden layer
        '''

        # create the layers for the model
        layers_list = [("input layer", nn.Linear(state_size, layer_size)), ("input layer RELU", nn.ReLU())]
        for i in range(hidden_amount):
            layers_list.append((f"hidden layer ({i})", nn.Linear(layer_size, layer_size)))
            layers_list.append((f"hidden layer ({i}) RELU", nn.ReLU()))
        layers_list.append(("output layer", nn.Linear(layer_size, 1)))

        # create the model itself
        self.model = nn.Sequential(
            OrderedDict(
                layers_list
            )
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def forward(self, state: torch.Tensor):
        return self.model(state)
    
    def loss(self, states: torch.Tensor, rewards: torch.Tensor):
        '''Calculates the loss of the value function'''
        target_returns = torch.flip(torch.cumsum(torch.flip(rewards, dims=(0,)), dim=0), dims=(0,)).detach()

        loss_val = torch.mean((self.forward(states).squeeze() - target_returns)**2)

        return loss_val

    def optimizer_step(self, states: torch.Tensor, rewards: torch.Tensor):
        loss = self.loss(states, rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
