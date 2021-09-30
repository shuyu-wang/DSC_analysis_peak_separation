import numpy as np

# import own module
from LocationScaleProbability import LSPD


###### Standard Gaussian model ######
class Model():

    # Negative log likelihood function
    def negLogDen(self, x):
        return x**2/2 + np.log(2*np.pi) / 2

    # Differentiation of the negative log likelihood function
    def gradX(self, x):
        return x

    # The second derivative of the negative log likelihood function
    def gradX2(self, x):
        return np.ones(x.shape)

    def gradient(self, x):
        return self

    def laplacian(self, x):
        return self

    def scaledGradient(self, x, d = 1e-12):
        return self

    def print(self):
        return None

    def isValid(self):
        return True

    def makeValid(self):
        return self

    def assign(self, other):
        return None

    def genSamples(self, size = 1):
        return np.random.standard_normal(size)

    # Define operators on standard normal distribution
    def __add__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __mul__(self, scalar):
        return self

    def __truediv__(self, other):
        return self

    def norm(self):
        return 0

    # Functions regarding the standard normal distribution
    def density(self, x):
        # Standard normal distribution probability density function
        return 1/(np.sqrt(2*np.pi)) * np.exp( - x**2 / 2 )

    def denGradX(self, x):
        # First-order derivative of likelihood function
        return -x * self.density(x)

    def denGradX2(self, x):
        # Second-order derivative of the likelihood function
        return - self.density(x) + (-x * self.denGradX(x))

###### Gaussian Distribution ######
class Gauss( LSPD ):

    def __init__(self, m = 0, s = 1, optM = False, optS = False):
        self.std = Model()
        self.m = m
        self.s = s

        self.optM = optM
        self.optS = optS




       