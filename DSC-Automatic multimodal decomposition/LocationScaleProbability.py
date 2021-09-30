import numpy as np
import numpy.linalg as la

# import own module
import GradientDescent as GD


###### Standear location scale probability distribution ######
class LSPD:
    
    # Subclasses for density and nll for LSFD and GM
    def __init__(self, std, m = 0, s = 1, optM = False, optS = False):
        self.std = std  # standard distribution 
        self.m = m  # location parameter
        self.s = s  # scale parameter

        # Boolean variables tracking which varialbes are optimized
        # Optimized logo,, boolen variables
        self.optM = optM
        self.optS = optS

    def print(self):
        print('m = ' + str(self.m))
        print('optM =' + str(self.optM))
        print('s = ' + str(self.s))
        print('optS =' + str(self.optS))
        self.std.print()

    # checks if the distribution is valid
    def isValid(self):
        if np.all(self.s > 0) and self.std.isValid():
            return True
        else:
            return False

    def makeValid(self, thres = 1e-6):
        self.std.makeValid()
        self.s = np.maximum(thres, self.s) # comparison of parameters
        return self

    # Assign the variables of another LSPDamily object to the current one
    def assign(self, other):
        self.std.assign(other.std)
        self.m = other.m
        self.s = other.s

    # Define operators on location scale family
    def __add__(self, other):
        return LSPD(self.std + other.std, self.m + other.m, self.s + other.s, self.optM, self.optS)

    def __sub__(self, other):
        return LSPD(self.std - other.std, self.m - other.m, self.s - other.s, self.optM, self.optS)

    # Ideally would have scalar and elementwise multiplication
    def __mul__(self, scalar):
        return LSPD(self.std * scalar, self.m * scalar, self.s * scalar, self.optM, self.optS)

    def __truediv__(self, other):
        return LSPD(self.std / other.std, self.m / other.m, self.s / other.s, self.optM, self.optS)

    def norm(self):
        return np.sqrt(self.std.norm()**2 + la.norm(self.m)**2 + la.norm(self.s)**2)

    def getMS(self):
        return self.m, self.s

    def setMS(self, m, s):
        self.m = m
        self.s = s

    def setM(self, m):
        self.m = m

    def getM(self):
        return self.m

    def setS(self, s):
        self.s = s

    def getS(self):
        return self.s

    def setOptM(self, optM):
        self.optM = optM

    def setOptS(self, optS):
        self.optS = optS

    def setOpt(self, optM, optS):
        self.optM = optM
        self.optS = optS

    # Generate location scale distribution sample
    def genSamples(self, size = 1):
        if size > 1:
            return self.s * self.std.genSamples(size) + self.m  
        elif size == 1:
            return self.s * self.std.genSamples(self.m.shape) + self.m

    # Negative logarithmic density function
    def negLogDen(self, x):
        m, s = self.getMS()
        return self.std.negLogDen((x-m)/s) + np.log(s)

    # The first and second order derivation of probability density function
    def gradM(self, x):
        m, s = self.getMS()
        return -1/s * self.std.gradX((x-m)/s)

    def gradM2(self, x):
        m, s = self.getMS()
        return 1/s**2 * self.std.gradX2((x-m)/s)

    def gradS(self, x):
        m, s = self.getMS()
        xm = (x-m)/s
        return self.std.gradX(xm) * -xm/s + 1/s

    def gradS2(self, x):
        m, s = self.getMS()
        xm = (x-m)/s
        return (self.std.gradX2(xm) * (xm/s)**2 + self.std.gradX(xm) * 2*xm/s**2) - 1/s**2

    # The first and second order derivation of probability density function
    def gradX(self, x):
        m, s = self.getMS()
        return 1/s * self.std.gradX((x-m)/s)

    def gradX2(self, x):
        m, s = self.getMS()
        return 1/s**2 * self.std.gradX2((x-m)/s) 


    # Use the pipeline function to move the sum
    def gradient(self, x):
        gradM = np.sum(self.gradM(x)) if self.optM else 0   
        gradS = np.sum(self.gradS(x)) if self.optS else 0
        return LSPD(self.std.gradient(x), gradM, gradS, self.optM, self.optS)

    def laplacian(self, x):
        gradM2 = np.sum(self.gradM2(x)) if self.optM else 0
        gradS2 = np.sum(self.gradS2(x)) if self.optS else 0
        return LSPD(self.std.laplacian(x), gradM2, gradS2, self.optM, self.optS)

    def scaledGradient(self, x, d = 1e-12):
        gradM = np.sum(self.gradM(x)) / np.sum(self.gradM2(x)) if self.optM else 0
        gradS = np.sum(self.gradS(x)) / (abs(np.sum(self.gradS2(x))) + d) if self.optS else 0
        # in log domain, have to scale gradS by exp(log(s)) = s
        return LSPD(self.std.scaledGradient(x), gradM, gradS, self.optM, self.optS)


    # Negative Log Likelihood 
    def negLogLike(self, x):
        return np.sum(self.negLogDen(x))



    # Parameter Estimation
    # Optimize parameters. Optimize parameters together according to data x
    def optimize(self, x, maxIter = 32, plot = False):
        
        # defineOptimizationParameters() method from GD.py
        params = GD.defineOptimizationParameters(maxIter = maxIter, minDecrease = 1e-5)
        obj = lambda E : E.negLogLike(x)
        grad = lambda E : E.scaledGradient(x)
        updateVariables = lambda E, dE, s : E - (dE * s)
        projection = lambda E : E.makeValid()
        E, normArr, stepArr = GD.gradientDescent(self, obj, grad, projection, updateVariables, params)
        self.assign(E)
        if plot:
            import matplotlib.pyplot as plt
            plt.subplot(121)
            plt.plot(normArr)
            plt.subplot(122)
            plt.plot(stepArr)
            plt.show()
        return self

    
    # Density 
    def density(self, x):
        return np.exp(-self.negLogDen(x))

    # Compute gradient of density given gradients of negative log-density
    def denGrad(self, den, nllGrad):
        return den * -nllGrad

    def denGrad2(self, den, nllGrad, nllGrad2):
        return den * (nllGrad**2 - nllGrad2)

    def denGradX(self, x):
        return self.denGrad(self.density(x), self.gradX(x))

    def denGradX2(self, x):
        return self.denGrad2(self.density(x), self.gradX(x), self.gradX2(x))

    def denGradM(self, x):
        return self.denGrad(self.density(x), self.gradM(x))
        # return self.density(x) * -self.gradM(x)

    def denGradM2(self, x):
        return self.denGrad2(self.density(x), self.gradM(x), self.gradM2(x))

    def denGradS(self, x):
        return self.denGrad(self.density(x), self.gradS(x))

    def denGradS2(self, x):
        return self.denGrad2(self.density(x), self.gradS(x), self.gradS2(x))


