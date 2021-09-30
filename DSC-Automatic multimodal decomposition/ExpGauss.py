import numpy as np
import numpy.linalg as la
from scipy.stats import skew, moment, norm
from scipy.special import erfcx, erfc

# import own module
import GradientDescent as GD
from LocationScaleProbability import LSPD
from SpecialFunctions import log_erfc


###### Standard Exponential modified Gaussian(ExpG) model ######
class Model:

    # standard parameterization of ExpG model
    def __init__(self, a = 1, optA = False):
        self.a = a  
        # Tracking whether or not a is optimized, boolean 
        self.optA = optA   


    # Check 
    def setA(self, a):
        self.a = a

    def getA(self):
        return self.a

    def setOptA(self, optA):
        self.optA = optA

    def getOptA(self):
        return self.optA

    # Define operators on ExpG distribution
    def __add__(self, other):
        return Model(self.a + other.a, self.optA)

    def __sub__(self, other):
        return Model(self.a - other.a, self.optA)

    def __mul__(self, scalar):
        return Model(self.a * scalar, self.optA)

    def __truediv__(self, other):
        return Model(self.a / other.a, self.optA)

    def norm(self):
        return la.norm(self.a)

    def print(self):
        print('a = ' + str(self.a))
        print('optA = ' + str(self.optA))

    def isValid(self):
        return True if self.a > 0 else False

    def makeValid(self, thres = 1e-6):
        self.a = max(self.a, thres)

    def assign(self, other):
        self.a = other.a

    # Generate a standard normal distribution array + exponential distribution array = standard ExpG distributed
    def genSamples(self, size = 1):
        return np.random.standard_normal(size) + np.random.exponential(scale = 1/self.a, size = size)

    # Negative log likelihood function. Standard Exponential Modified Gaussian Distribution
    def negLogDen(self, x):
        a = self.a
        nld = -np.log(a/2) - a**2 / 2 + a*x - log_erfc( (a-x) / np.sqrt(2) )
        return nld

    def density(self, x):
        # Likelihood function
        return np.exp(-self.negLogDen(x))

    def isConvexInA(self):
        return (True if self.a < 1 else False)

    def _get_d(self, x):
        a = self.a
        d = (a-x) / np.sqrt(2)
        return d

    def _get_de(self, x):
        d = self._get_d(x)
        e = 1 / erfcx(d)
        return d, e

    # First derivative of negative log density w.r.t. x
    def gradX(self, x):
        a = self.a
        d, e = self._get_de(x)
        return a - np.sqrt(2/np.pi) * e

    # Second derivative of standard negative log density w.r.t. x
    def gradX2(self, x):
        a = self.a
        d, e = self._get_de(x)
        return (2/np.pi * e**2 - 2/np.sqrt(np.pi) * e * d)

    # First derivative of standard negative log density w.r.t. a
    # a: alpha scalar
    def gradA(self, x):
        a = self.a
        d, e = self._get_de(x)
        return -(1/a + a) + x + np.sqrt(2/np.pi) * e

    # Second derivative of standard negative log density w.r.t. a
    def gradA2(self, x):
        a = self.a
        d, e = self._get_de(x)
        return (1/a**2 - 1) + 2/np.pi * e**2 - 2/np.sqrt(np.pi) * e * d

    # Gradient descent. Distribution parameter a
    def gradient(self, x):
        return Model(np.sum(self.gradA(x)) if self.optA else 0)

    # On the value of the second derivative
    def laplacian(self, x):
        return Model(np.sum(self.gradA2(x)) if self.optA else 0)

    def scaledGradient(self, x, d = 1e-12):
        return Model(np.sum(self.gradA(x)) / (abs(np.sum(self.gradA2(x)) + d)) if self.optA else 0)

# The locationScaleFamily.py module was introduced. From the Gaussian 
# exponential model of the standard distribution to the exponential modified Gaussian model.
class ExpG( LSPD ):
    
    def __init__(self, a = 1, m = 0, s = 1, optA = False, optM = False, optS = False):
        self.m = m  # location parameter
        self.s = s  # scale parameter
        self.std = Model(a, optA)  # the standard distribution on which we are basing the LSPDamily

        # Boolean variables tracking which varialbes are optimized
        self.optM = optM
        self.optS = optS

    # getMS() comes from the LSPD module
    def getAMS(self):
        return self.std.a, self.getMS()

    def setAMS(self, a, m, s):
        self.std.setA(a)
        self.m = m
        self.s = s

    def setOpt(self, optA, optM, optS):
        self.std.setOptA(optA)
        self.optM = optM
        self.optS = optS

    def setA(self, a):
        self.std.setA(a)

    def setOptA(self, optA):
        self.std.setOptA(optA)

    def gradA(self, x):
        m, s = self.getMS()
        return self.std.gradA((x-m)/s)

    def gradA2(self, x):
        m, s = self.getMS()
        return self.std.gradA2((x-m)/s)


###### demo ######
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    E = ExpG()
    E.print()
    n = 1024
    x = E.genSamples(n)
    '''
    import matplotlib.pyplot as plt 
    dom = np.linspace(-1, 1, n)
    plt.plot(dom, x)
    plt.show()
    '''
    E.setAMS(.5, 0, 1)
    E.print()

    E.setOptA(True)
    E.setOptM(False)
    E.setOptS(False)
    '''
    E.setOptA(True)
    E.setOptM(True)
    E.setOptS(True)
    '''
    E.print()
    E.optimize(x)
    E.print()
