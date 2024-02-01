from scipy.integrate import quad
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def density(alpha, beta, t):
    return (
        beta
        / alpha
        * (t / alpha) ** (beta - 1)
        * np.exp(-((t / alpha) ** beta))
    )


def distribution(alpha, beta, t):
    return quad(density, 0.2, t, args=(alpha, beta))[0]


SCALAR = MinMaxScaler(feature_range=(0.2, 4))
