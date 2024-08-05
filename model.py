from torch import nn

# Define the network architecture
class TwoLayerNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, leaky_param: float=0.1):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.leakyrelu = nn.LeakyReLU(leaky_param)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.leakyrelu(out)
        out = self.fc2(out)
        return out
