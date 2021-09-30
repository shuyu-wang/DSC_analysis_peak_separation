"""  Identify Peak Signal  """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal
from matplotlib import pyplot
from scipy.optimize import curve_fit
#%matplotlib inline

import peakutils
from peakutils.plot import plot as pplot


#file = r'../combined_dsc1.csv' # BSA protein
#file = r'../calfitter_dsc2.csv' # LinB protein
#file = r'../lysozyme_dsc1.csv' # Lysozyme protein
#file = r'../dvd_dsc1.csv' # dvd protein
file = r'../dvd_dsc2.csv' # dvd protein, subtract buffer
#file = r'../dvd_dsc3.csv' # dvd protein, subtract buffer  
#file = r'../mab_dsc1.csv' # mab protein

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
y = I[i]
pyplot.figure(figsize=(10,6))
pyplot.plot(x, y)
plt.tick_params(labelsize=16)
pyplot.title("Raw Data", size=22)


# Getting a first estimate of the peaks
# By using peakutils.indexes, we can get the indexes of the peaks from the data. 
# Due to noise, it will be just a rough approximation.
indexes = peakutils.indexes(y, thres=0.5, min_dist=10)
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
            guess[i] = a
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

# Try to automate peak deconvolution
def AutoMulPeakDecom(x, y):
    indexes = peakutils.indexes(y, thres=0.5, min_dist=10)
    cen = x[indexes]
    num = len(cen)*3
    amp=0.02
    wid =10
    
    # Or add regularization here
    #guess = Mul_Param(n = Num, a = AmpMin, m = Cen, s = WidMin)
    guess = Mul_Param(n = num, a = amp, m = cen, s = wid)
    popt, pcov = curve_fit(Mul_Gaussian, x, y, guess, maxfev=200000)
    
    print(popt)
    #print(pcov)
    # Gaussian model parameters
    print("[[Variables]]")
    for i in range(0, num, 3): # range(n) gaussian model number
        '''
        print("Model Variables", i, ":")
        print("  amp", i, ":", popt[i])
        print("  cen", i, ":", popt[i+1])
        print("  wid", i, ":", popt[i+2])
        '''
        print("Model Variables", i, ":")
        print("  amp", i, ":", popt[i])
        print("  cen", i, ":", popt[i+1])
        print("  wid", i, ":", popt[i+2])
        
    # Plot peak fitting
    plt.figure(figsize=(8,6))
    plt.plot(x, y, "C2.", label='Net Signal')
    plt.plot(x, Mul_Gaussian(x,*popt), 'C3-', label='Gaussian Fitting')
    
    g1 = Gaussian(x, popt[0], popt[1], popt[2]);
    g2 = Gaussian(x, popt[3], popt[4], popt[5]);
    g3 = Gaussian(x, popt[6], popt[7], popt[8]);
    g4 = Gaussian(x, popt[9], popt[10], popt[11]);
    plt.plot(x, g1+g2+g3+g4, label='Gaussian Fitting')
    plt.legend(loc='best', fontsize=16)
    plt.show() 

AutoMulPeakDecom(x, y)














