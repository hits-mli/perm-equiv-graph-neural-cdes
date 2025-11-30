import torch
import torch.nn as nn


class GeneDynamics(nn.Module):
    def __init__(self, A, b, f=1, h=2):
        super(GeneDynamics, self).__init__()
        self.A = A  # Adjacency matrix
        self.b = b
        self.f = f
        self.h = h

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dxi(t)/dt = -b*xi^f + \sum_{j=1}^{N}Aij xj^h / (1 + xj^h)
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        if hasattr(self.A, "is_sparse") and self.A.is_sparse:
            f = -self.b * (x**self.f) + (self.A @ (x**self.h)) / (x**self.h + 1)
        else:
            f = -self.b * (x**self.f) + (self.A @ (x**self.h)) / (x**self.h + 1)
        return f
