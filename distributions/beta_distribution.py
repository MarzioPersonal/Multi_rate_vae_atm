import torch
import torch.nn as nn
from torch.distributions import Uniform
import numpy as np


class BetaUniform(Uniform):
    @property
    def mean(self):
        return super().mean

    @property
    def mode(self):
        return super().mode

    @property
    def stddev(self):
        return super().stddev

    @property
    def variance(self):
        return super().variance

    def __init__(self, low=0.01, high=10., validate_args=None):
        super().__init__(np.log(low), np.log(high), validate_args)

    def expand(self, batch_shape, _instance=None):
        return super().expand(batch_shape, _instance)

    def support(self):
        return super().support()

    def rsample(self, sample_shape=torch.Size()):
        return super().rsample(sample_shape)

    def log_prob(self, value):
        return super().log_prob(value)

    def cdf(self, value):
        return super().cdf(value)

    def icdf(self, value):
        return super().icdf(value)

    def entropy(self):
        return super().entropy()
