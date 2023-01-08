import torch
from torch import nn



class DropEdge(nn.Module):
    '''
    Implimenting DropEdge https://openreview.net/forum?id=Hkx1qkrKPr
    '''

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, ei, ew=None):
        if self.training and self.p > 0:
            mask = torch.rand(ei.size(1))
            if ew is None:
                return ei[:, mask > self.p]
            else:
                return ei[:, mask > self.p], ew[mask > self.p]

        if ew is None:
            return ei
        else:
            return ei, ew