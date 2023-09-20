import torch


class Distribution:
    def __init__(self, tensor):
        self.tensor = torch.clone(tensor.detach())

    def get_values(self):
        return self.tensor.cpu().flatten()
