import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """ Basic neural network used as a approximator"""

    def __init__(self, inputs_no: int, output_no: int, discrete_outputs: bool):
        super().__init__()
        self.discrete_outputs = discrete_outputs
        self.fc1 = nn.Linear(inputs_no, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_no)
        self.dropout = nn.Dropout(p=0.8)

    def forward(self, x):
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        if self.discrete_outputs:
            x = F.softmax(x, dim=1)

        return x


class ComplexNet(nn.Module):
    """ More complex neural network used as a approximator"""

    def __init__(self, inputs_no: int, output_no: int, discrete_outputs: bool):
        super().__init__()
        self.discrete_outputs = discrete_outputs
        self.fc1 = nn.Linear(inputs_no, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_no)
        self.dropout = nn.Dropout(p=0.8)

    def forward(self, x):
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        if self.discrete_outputs:
            x = F.softmax(x, dim=1)

        return x
