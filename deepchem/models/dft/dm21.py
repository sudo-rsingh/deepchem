import torch.nn as nn
import torch


class dm21(nn.Module):
    def __init__(self):
        super(dm21, self).__init__()
        self.lin_tanh = nn.Linear(11, 256)
        self.lin_elu = nn.ModuleList()
        for i in range(6):
            self.lin_elu.append(nn.Linear(256, 256))
        self.final = nn.Linear(256, 3)
        self.acti_tanh = nn.Tanh()
        self.acti_elu = nn.ELU()
        self.acti_scaled_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.log(torch.abs(x) + torch.tensor([10**-4], device=x.device))
        x = self.acti_tanh(self.lin_tanh(x))
        for i in range(6):
            x = self.acti_elu(self.lin_elu[i](x))
        x = self.acti_scaled_sigmoid(self.final(x))
        return x
