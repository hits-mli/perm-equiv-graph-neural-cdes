import torch
import torch.nn as nn


class MutualDynamics(nn.Module):
    #  dx/dt = b +
    def __init__(self, A, b=0.1, k=5.0, c=1.0, d=5.0, e=0.9, h=0.1):
        super(MutualDynamics, self).__init__()
        self.A = A  # Adjacency matrix, symmetric
        self.b = b
        self.k = k
        self.c = c
        self.d = d
        self.e = e
        self.h = h

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dxi(t)/dt = bi + xi(1-xi/ki)(xi/ci-1) + \sum_{j=1}^{N}Aij *xi *xj/(di +ei*xi + hi*xj)
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        n, d = x.shape
        f = self.b + x * (1 - x / self.k) * (x / self.c - 1)
        if d == 1:
            # one 1 dim can be computed by matrix form
            if hasattr(self.A, "is_sparse") and self.A.is_sparse:
                outer = torch.sparse.mm(
                    self.A,
                    torch.mm(x, x.t())
                    / (
                        self.d
                        + (self.e * x).repeat(1, n)
                        + (self.h * x.t()).repeat(n, 1)
                    ),
                )
            else:
                outer = torch.mm(
                    self.A,
                    torch.mm(x, x.t())
                    / (
                        self.d
                        + (self.e * x).repeat(1, n)
                        + (self.h * x.t()).repeat(n, 1)
                    ),
                )
            f += torch.diag(outer).view(-1, 1)
        else:
            # high dim feature, slow iteration
            if hasattr(self.A, "is_sparse") and self.A.is_sparse:
                vindex = self.A._indices().t()
                for k in range(self.A._values().__len__()):
                    i = vindex[k, 0]
                    j = vindex[k, 1]
                    aij = self.A._values()[k]
                    f[i] += (
                        aij * (x[i] * x[j]) / (self.d + self.e * x[i] + self.h * x[j])
                    )
            else:
                vindex = self.A.nonzero()
                for index in vindex:
                    i = index[0]
                    j = index[1]
                    f[i] += (
                        self.A[i, j]
                        * (x[i] * x[j])
                        / (self.d + self.e * x[i] + self.h * x[j])
                    )
        return f
