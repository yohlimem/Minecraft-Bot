from collections import OrderedDict
from torch import nn
from torch import optim
import torch
from state_value import StateValue

class Policy(nn.Module):
    def __init__(self, value_function: StateValue, state_size: int, actions_size: int, hidden_amount: int =3, layer_size: int =256):
        super().__init__()

        '''
        state_size size is a number bigger than 1 that represents the amount of inputs for the network
        actions_size is a number bigger than 1 that represents the amount of outputs of the network
        hidden_amount is a number that represents the amount of hidden layers
        layer_size is a number bigger than 1 that represents the amount of neurons in a hidden layer
        '''

        # create the layers for the model
        layers_list = [("input layer", nn.Linear(state_size, layer_size)), ("input layer RELU", nn.ReLU())]
        for i in range(hidden_amount):
            layers_list.append((f"hidden layer ({i})", nn.Linear(layer_size, layer_size)))
            layers_list.append((f"hidden layer ({i}) RELU", nn.ReLU()))
        layers_list.append(("output layer", nn.Linear(layer_size, actions_size)))
        layers_list.append(("output layer sofmax", nn.Softmax(dim=1)))

        # create the model itself
        self.model = nn.Sequential(
            OrderedDict(
                layers_list
            )
        )

        self.value_function = value_function

        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def forward(self, state: torch.Tensor):
        return self.model(state)
    
    def objective(self, states: torch.Tensor, actions_taken_indices: torch.Tensor, old_probs: torch.Tensor, rewards: torch.Tensor, advantages: torch.Tensor):
        new_probs = self.forward(states).gather(1, actions_taken_indices.long().unsqueeze(1)).squeeze() # uses the action indexes to find the correct probabilties

        ratio = new_probs/old_probs
        biggest_grad_variance = 0.1 # epsilon

        return -torch.mean(torch.min(ratio*advantages, torch.clip(ratio, 1-biggest_grad_variance, 1+biggest_grad_variance)*advantages)) # minus for maximizing in the grad direction instead of minimizing
    
    def advantage(self, states: torch.Tensor, rewards: torch.Tensor):
        with torch.no_grad():
            target_returns = torch.flip(torch.cumsum(torch.flip(rewards, dims=(0,)), dim=0), dims=(0,))
            adv = target_returns - self.value_function(states).squeeze()
            return (adv - adv.mean()) / (adv.std() + 1e-8) # Standardization
        
    def optimizer_step(self, states: torch.Tensor, actions_taken_indices: torch.Tensor, old_probs: torch.Tensor, rewards: torch.Tensor, advantages: torch.Tensor):
        objective = self.objective(states, actions_taken_indices, old_probs, rewards, advantages)
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        
        self.value_function.optimizer_step(states, rewards)