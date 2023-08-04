import torch
from utils.optimizer.sam import SAM

class AdamWithMomentum(torch.optim.Adam):
    def __init__(self, params, lr=0.001, momentum=0.9):
        super(AdamWithMomentum, self).__init__(params, lr=lr)
        self.momentum = momentum