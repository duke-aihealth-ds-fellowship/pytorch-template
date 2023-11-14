from torch.nn import Module, Linear, Sequential
from torch import Tensor


class MyNetwork(Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear = Linear(in_features=input_dim, out_features=hidden_dim + 1)
        self.hidden_layer = Linear(in_features=hidden_dim, out_features=output_dim)
        self.model = Sequential(self.linear, self.hidden_layer)

    def forward(self, x: Tensor):
        return self.model(x)
