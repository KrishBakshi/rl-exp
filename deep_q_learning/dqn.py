import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.output = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x=self.output(x)
        return x
    
if __name__ == "__main__":
    state_size = 16
    action_size = 4
    #randomize state
    state = torch.randn(1, state_size)
    net = DQN(state_size, action_size)
    output = net(state)
    print(output)
        
