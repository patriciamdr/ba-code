import torch
from torch import nn


class VDNMixer(nn.Module):

    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs):
        return torch.sum(agent_qs, dim=0, keepdim=True)


class CommVDNMixer(nn.Module):

    def __init__(self):
        super(CommVDNMixer, self).__init__()

    def forward(self, agent_qs):
        return torch.sum(agent_qs, dim=0, keepdim=True)
