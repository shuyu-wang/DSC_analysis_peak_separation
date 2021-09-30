import numpy as np
import numpy.linalg as la
from copy import deepcopy

# import own module
import GaussModel as Gauss
import ExpGauss as ExpG
from LocationScaleProbability import LSPD


###### Standard Exponential modified Gaussian Mixture (ExpGMix) model ######
class Model():

    def __init__(self, a = 1, z = 0, optA = False, optZ = False):
        self.ExpG = ExpG.Model(a, optA)
        self.Gauss = Gauss.Model()
        # Indicator function
        self.z = z  
        # Tracking whether or not a is optimized, boolean 
        self.optZ = optZ 

    def print(self):
        self.ExpG.print()
        print('z = ' + str(self.z))
        print('optZ = ' + str(self.optZ))

    def getZ(self):
        return self.z

    def setZ(self, z):
        self.z = z

    def getAZ(self):
        return self.a, self.z

    def setAZ(self, a, z):
        self.ExpG.setA(a)
        self.z = z

    def setOpt(self, optA, optZ):
        self.ExpG.optA = optA
        self.optZ = optZ

    # Check
    def isValid(self):
        if self.ExpG.isValid() and self.Gauss.isValid():
            return True
        else:
            return False

    def makeValid(self, thres = 1e-6):
        self.ExpG.makeValid()
        self.Gauss.makeValid()
        return self

    # Assign the variables of another LSPDamily object to the current one
    def assign(self, other):
        self.ExpG.assign(other.ExpG)
        self.z = other.z

    # Define operators on standard ExpGaussMix objects
    def __add__(self, other):
        return Model(self.ExpG.getA() + other.ExpG.getA(), self.z + other.z, self.ExpG.getOptA(), self.optZ)

    def __sub__(self, other):
        return Model(self.ExpG.getA() - other.ExpG.getA(), self.z - other.z, self.ExpG.getOptA(), self.optZ)

    # Ideally would have scalar and elementwise multiplication
    def __mul__(self, scalar):
        return Model(self.ExpG.getA() * scalar, self.z * scalar, self.ExpG.getOptA(), self.optZ)

    def __truediv__(self, other):
        return Model(self.ExpG.getA() / other.ExpG.getA(), self.z / other.z, self.ExpG.getOptA(), self.optZ)

    # Calculate the norm of the object
    def norm(self):
        return np.sqrt(self.ExpG.norm()**2 + la.norm(self.z)**2)

# Calculate the derivative of the negative log-likelihood function of the ExpGaussMix distribution, 
# which is convenient for gradient calculation to solve the minimum value. That is, the probability is the greatest.
    def negLogDen(self, x):
        z = self.getZ()
        return (1-z) * self.Gauss.negLogDen(x) + z * self.ExpG.negLogDen(x)
    
    def density(self, x):
        return np.exp(-self.negLogDen(x))

    def gradX(self, x):
        z = self.getZ()
        return (1-z) * self.Gauss.gradX(x) + z * self.ExpG.gradX(x)

    def gradX2(self, x):
        z = self.getZ()
        return (1-z) * self.Gauss.gradX2(x) + z * self.ExpG.gradX2(x)

    def gradA(self, x):
        z = self.getZ()
        return z * self.ExpG.gradA(x)

    def gradA2(self, x):
        z = self.getZ()
        return z * self.ExpG.gradA2(x)

    def gradient(self, x):
        return Model(np.sum(self.gradA(x)) if self.ExpG.getOptA() else 0, 0)

    def laplacian(self, x):
        return Model(np.sum(self.gradA2(x)) if self.ExpG.getOptA() else 0, 0)

    def scaledGradient(self, x, d = 1e-12):
        return Model(np.sum(self.gradA(x)) / (abs(np.sum(self.gradA2(x)) + d)) if self.ExpG.getOptA() else 0, 0)

    # Generate a standard ExpG mixture distribution
    def genSamples(self, size = 1):
        z = self.getZ()
        ind = np.random.random(size) < z
        return (1-ind) * self.Gauss.genSamples(size) + ind * self.ExpG.genSamples(size)

    # Calculate the expected value of z, given mixture probabilities mix
    def expectedZ(self, x, mix):

        # compute responsibilities
        GauDen = self.Gauss.density(x)
        ExpGDen = self.ExpG.density(x)

        # Return (mix * ExpGDen) / ((1-mix) * GauDen + mix * ExpGDen)
        # for numerical stability
        ind = (mix*ExpGDen + (1-mix)*GauDen) == 0
        notInd = np.logical_not(ind)
        z = np.zeros(x.shape)
        z[ notInd ] = (mix * ExpGDen[notInd]) / (mix * ExpGDen[notInd] + (1-mix) * GauDen[notInd])
        z[ np.logical_and(ind, x>0) ] = 1
        z[ np.logical_and(ind, x<0) ] = 0
        return z

###### Exponential modified Gaussian mixture model ######
class ExpGaussMix( LSPD ):


    # could be a child of an abstract mixture model class
    def __init__(self, a = 1, m = 0, s = 1, z = 0, optA = False, optM = False, optS = False, optZ = False):

        self.std = Model(a, z, optA, optZ)
        self.m = m   # location parameter
        self.s = s   # scale parameter

        self.optM = optM
        self.optS = optS

    def getAMSZ(self):
        a, z = self.std.getAZ()
        return a, self.getMS(), z

     # Set variables, makes sure that both distributions are updated
    def setAMSZ(self, a, m, s, z):
        self.std.setAZ(a, z)
        self.setMS(m, s)

    def getZ(self):
        return self.std.getZ()

    def getOptZ(self):
        return self.std.optZ

    def setZ(self, z):
        return self.std.setZ(z)

    # setting all optimization indicators
    def setOpt(self, optA, optM, optS, optZ):
        self.std.setOpt(optA, optZ)
        self.setOptM(optM)
        self.setOptS(optS)



    ####### Learning ######
    def calculateMix(self):
        return np.mean(self.getZ())

    def expectedZ(self, x, mix):
        m, s = self.getMS()
        return self.std.expectedZ( (x-m)/s, mix)   

    def expectationStep(self, x, mix):  
        self.setZ(self.getOptZ() * self.expectedZ(x, mix))  
        return self

    def maximizationStep(self, x, mix, optMix, maxIter):

        # optimize continuous parameters of ExpGaussMix model
        super(ExpGaussMix, self).optimize(x, maxIter = maxIter)

        # optimize mixture coefficient  
        if optMix:
            mix = np.mean(self.getZ())  

        return mix

###### EM algorithm learning Exp correction Gaussian model mixture, peak signal intensity distribution ######
    def optimize(self, x, mix = 1/2, optMix = True, maxIter = 8, minChange = 1e-6, maxMaxIter = 128):
        converged = False
        iter = 0
        while not converged:

            oldSelf = deepcopy(self)

            mix = self.maximizationStep(x, mix, optMix, maxMaxIter)

            if np.any(self.getOptZ()):
                self.expectationStep(x, mix)

            iter += 1
            if iter > maxIter or (oldSelf-self).norm() < minChange:
                converged = True
        '''
        print("location parameter：m=",self.m)
        print("scale parameter：s=",self.s)
        print("exponentially：a=",self.std.EMG.a) # Added by myself
        '''
        
        return self


###### Sample probability signal distribution demonstration ######
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    #file = r'../combined_dsc1.csv' # BSA dsc data
    #file = r'../calfitter_dsc2.csv'# LinB dsc data, Calfitter
    #file = r'../lysozyme_dsc1.csv' # lysozyme dsc data
    
    ######## second work ##########
    file = r'../dvd_dsc1.csv' # dvd protein
    #file = r'../dvd_dsc2.csv' # dvd protein, subtract buffer
    #file = r'../mab_dsc1.csv' # mab protein
    #file = r'../mab696_dsc2.csv' # mab protein. substract baseline
    
    
    df = pd.read_csv(file, header = None)
    df = df.T
    
    Q = df.iloc[0,1:]
    Q = Q.astype(float)
    Q = np.array(Q,dtype=np.float32)
    
    I = df.iloc[1:, 1:]
    #I = abs(I.astype(float))
    I = I.astype(float)
    I =np.array(I,dtype=np.float32)

    x = I
    
    # Lysozyme, normalization
    #x = -I
    #x = x * 17160 
    
    # BSA, normaliztion
    #x = I * 687.70764 
    
    # Mab696, normalizition
    x = I + 1

    mix=0.5
    ParVariable = ExpGaussMix(a=1,z=mix)
    dom=np.zeros(x.shape)
    GaussIsIn = np.ones(x.shape)
    for j in range(len(x)):
        ParVariable.setOpt(True, True, True, True)
        ParVariable.optimize(x[j],mix)
        ParChange = ParVariable.getZ()
        
        for k in range(len(ParChange)):
            #if ParChange[k] > mix+0.02 :  # Lysozyme dsc data
            #if ParChange[k] > mix+0.00578:
            if ParChange[k] > mix+0.1:
                #ParChange[k] = 1
                dom[j,k] = 1
        fig = plt.figure(figsize=(6,4))
        ax1 = fig.add_subplot(111)
        plot1 = ax1.plot(Q, x[j], c='C1', label='Measuring signal')
        ax2 = ax1.twinx()
        plot2 = ax2.plot(Q, dom[j], c='C0', label='Distribution probability')
        lines = plot1 + plot2
        ax1.legend(lines, [l.get_label() for l in lines])
        plt.legend(loc='best')
        ax1.set_xlabel("Temperature", size=18)
        ax1.set_ylabel("Heat capacity", size=18)
        ax2.set_ylabel("Distribution probability", size = 18)
        plt.title("scan=%s"%(j))
        plt.show()

    '''
    print('Inferred mixture probability:' + str(np.mean(ExpGaussMix.getZ())))
    print('True mixture probability:' + str(mix))
    ExpGaussMix.print()
    '''
      








