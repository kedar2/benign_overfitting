from torch import nn
import torch

# Define the network architecture
class TwoLayerNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, leaky_param: float=0.1):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc1.weight.data.normal_(0, 1 / (hidden_dim * input_dim ** 0.5))
        self.leakyrelu = nn.LeakyReLU(leaky_param)

        # second layer should be frozen, alternting between -1 and 1 in shape of hidden_dim
        self.fc2 = 2 * ((torch.ones(hidden_dim, 1, requires_grad=False) % 2) == 0).float() - 1

    def forward(self, x):
        out = self.fc1(x)
        out = self.leakyrelu(out)
        out = out @ self.fc2
        return out
