"""  Identify Peak Signal 
Use peak information as the initial value 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal
from matplotlib import pyplot
from scipy.optimize import curve_fit

import peakutils
from peakutils.plot import plot as pplot
from lmfit.models import ExponentialModel

#file = r'../combined_dsc1.csv' # BSA protein
#file = r'../calfitter_dsc2.csv' # LinB protein
#file = r'../lysozyme_dsc1.csv' # Lysozyme protein
file = r'../dvd_dsc2.csv' # dvd protein, subtract buffer
#file = r'../mab_dsc1.csv' # mab protein
#file = r'../mab696_dsc2.csv'

df = pd.read_csv(file, header = None)
df = df.T
#print(df)

Q = df.iloc[0,1:]
Q = Q.astype(float)
Q = np.array(Q,dtype=np.float32)
#print(Q)

I = df.iloc[1:, 1:]
I = I.astype(float)
I =np.array(I,dtype=np.float32)


# Preparing the data
i = 1
x = Q
#y =I[i]
#y = I[i]+abs(min(I[i]))  # abs(y)
y = I[i] + abs(min(I[i]))
pyplot.figure(figsize=(10,6))
pyplot.plot(x, y)
plt.tick_params(labelsize=16)
pyplot.title("Raw Data", size=22)


# Getting a first estimate of the peaks
# By using peakutils.indexes, we can get the indexes of the peaks from the data. 
# Due to noise, it will be just a rough approximation.
#indexes = peakutils.indexes(y, thres=0.5, min_dist=10) # DSC
indexes = peakutils.indexes(y, thres=0.3, min_dist=0.1) # mass spectra
print(indexes)
print(x[indexes], y[indexes])
pyplot.figure(figsize=(10,6))
pplot(x, y, indexes)
plt.tick_params(labelsize=16)
pyplot.title('Peak estimate', size=22)


# Gaussian distribution function
def Gaussian(x, a, m, s):
    
    #return a * np.exp( -((x-m)/s)**2 )
    return (a / (np.sqrt(2*np.pi) * s)) * np.exp(-(x-m)**2 / (2*s**2))

# Function expression of multiple Gaussian distributions
def Mul_Gaussian(x, *pars):
    
    params = np.array(pars)
    MulGauss = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp, cen, wid = params[i:i+3]
        MulGauss = MulGauss + Gaussian(x, amp, cen, wid)
    
    return MulGauss

# Parameters of multiple Gaussian distribution functions
def Mul_Param(n, a, m, s):
    
    guess = np.zeros(n)
    for i in range(0, n, 3):
        for j in range(0, len(m)):
            #guess[i] = a
            guess[i] = a[j]   #采用每个点对应的数值 
            guess[i+1] = m[j]
            guess[i+2] = s
    
    return guess

# Used to determine the initial value of Gaussian peak fitting. How to define a single value?
def GaussParam(ampWid):
    ampWid.sort()
    awLen = len(ampWid)
    if awLen < 2:
        return ampWid, 2*ampWid
    else:
        return ampWid[0], ampWid[-1]

# Root mean square error
def RMSE(target, prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
        
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))
    
    Rmse = np.sqrt(sum(squaredError) / len(squaredError))
    
    return Rmse

def AutoMulPeakDecom(x, y):
    #indexes = peakutils.indexes(y, thres=0.5, min_dist=10) # dsc data
    indexes = peakutils.indexes(y, thres=0.3, min_dist=10) # mass spectra
    #x[indexes], y[indexes]
    cen = x[indexes]
    num = len(cen)*3
    
    # dsc data. Test, want to added one sub-peak
    amp = 10
    wid = 310
    num = len(cen)*3 +2*3
    guess = Mul_Param(n = num, a = y[indexes], m = cen, s = wid)
    popt, pcov = curve_fit(Mul_Gaussian, x, y, guess, maxfev=300000)
    print(popt)
    #print(pcov)
    predictGauss = Mul_Gaussian(x,*popt)
    # Gaussian model parameters
    print("[[Variables]]")
    plt.figure(figsize=(8,6))
    j = 0
    for i in range(0, num, 3): # range(n) gaussian model number
        j +=1
        print("Model Variables", j, ":")
        print("  amp", j, ":", popt[i])
        print("  cen", j, ":", popt[i+1])
        print("  wid", j, ":", popt[i+2])
        plt.plot(x, Gaussian(x, popt[i], popt[i+1], popt[i+2]),  label= 'Gausssian')
    plt.plot(x, predictGauss, label='Gaussian Fitting')
    plt.legend(loc='best', fontsize=16)
    plt.show()
        
    # Plot peak fitting
    plt.figure(figsize=(8,6))
    plt.plot(x, y, "C2.", label='Net Signal')
    plt.plot(x, predictGauss, 'C3-', label='Gaussian Fitting')
    plt.legend(loc='best', fontsize=16)
    plt.show()
    
    # RMSE
    rmse = RMSE(y, predictGauss)
    print("\nRMSE:",rmse)

















