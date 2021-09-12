import torch
from torch.nn.modules.container import Sequential

# Define model
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack: Sequential = torch.nn.Sequential(
            torch.nn.Linear(28*28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10)
        )

    def forward(self, x) -> torch.nn.Sequential:
        x: torch.nn.Sequential = self.flatten(x)
        logits: torch.nn.Sequential = self.linear_relu_stack(x)
        return logits
