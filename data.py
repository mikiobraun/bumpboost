import numpy as np
from numpy.random import randn, rand

def sincdata(n=100, noise=0.1):
    x = rand(n) * 8 - 4
    y = np.sinc(x) + noise * randn(n)
    return x.reshape(-1, 1), y