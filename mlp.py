import torch.nn as nn
import torch.nn.functional as F
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer_size=64, output_size=[2,2,2,2]):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_layer_size)
        self.fc2 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.fc3 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.fc4 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.fc5 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)

        self.fc_binary = nn.Linear(self.hidden_layer_size, self.output_size[0])
        self.fc_ncu = nn.Linear(self.hidden_layer_size, self.output_size[1])
        self.fc_dir = nn.Linear(self.hidden_layer_size, self.output_size[2])
        self.fc_loc = nn.Linear(self.hidden_layer_size, self.output_size[3])

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)

        x_binary = self.fc_binary(x)
        x_ncu = self.fc_ncu(x)
        x_dir = self.fc_dir(x)
        x_loc = self.fc_loc(x)

        return x_binary, x_ncu, x_dir, x_loc