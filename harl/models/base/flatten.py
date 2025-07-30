from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)
