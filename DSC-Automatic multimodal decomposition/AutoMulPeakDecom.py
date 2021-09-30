"""
   Automatic multimodal decomposition. 
   Try to use automation to complete multi-peak decomposition. test-demo
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simps

# import modules published online
from lmfit.models import ExpressionModel
import peakutils
from peakutils.plot import plot as pplot

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


# Gaussian distribution function
def Gaussian(x, a, m, s):
    
    return a * np.exp( -((x-m)/s)**2 )

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
        guess[i] = a
        guess[i+1] = m
        guess[i+2] = s
    
    return guess
        
   
# Automatic multimodal decomposition
def AutoMulPeakDecom(x, y):
    
    print("***Peak Fitting......")
    x = np.array(x)
    y = np.array(y)
    
    # set parammers
    AmpMin = 1
    AmpMax = 1000
    CenMin = min(x)
    CenMax = max(x)
    WidMin = 0.1
    WidMax = 100   

    # Define the number of Gaussian models and multiple Gaussian functions
    Num = 2
    '''
    popt, pcov = curve_fit(func2, x, y, 
                           bounds=([AmpMin,AmpMin, CenMin,CenMin, WidMin,WidMin],
                                   [AmpMax,AmpMax, CenMax,CenMax, WidMax,WidMax]))
    ''' 
    guess = [AmpMin, CenMin, WidMin, AmpMin , CenMin, WidMin]
    popt, pcov = curve_fit(Mul_Gaussian, x, y, guess, maxfev=24000)
    
    # Gaussian model parameters
    print("[[Variables]]")
    for i in range(2): # range(n) gaussian model number
        print("Model Variables", i, ":")
        print("  amp", i, ":", popt[i])
        print("  cen", i, ":", popt[i+1])
        print("  wid", i, ":", popt[i+2])
    
    # Plot peak fitting
    plt.figure(dpi=600, figsize=(8,6))
    plt.plot(x, y, "C2.", label='Net Signal')
    plt.plot(x, Mul_Gaussian(x,*popt), 'C3-', label='Gaussian Fitting')
    plt.plot(x, Gaussian(x, popt[0], popt[1], popt[2]), 'C4--', label= 'Gausssian 1')
    plt.plot(x, Gaussian(x, popt[3], popt[4], popt[5]), 'C5--', label= 'Gausssian 2')
        
    plt.legend(loc='best', fontsize=16)
    plt.xlabel("Temperature (℃)", size=22, labelpad=10)
    plt.ylabel("Heat capacity (KJ/mol/K)", size=22, labelpad=10)
    plt.tick_params(labelsize=16)
    plt.show()
    
    # RMSE
    GauBestFit = Gaussian(x, popt[0], popt[2], popt[4]) + Gaussian(x, popt[1], popt[3], popt[5])
    Rmse = RMSE(y, GauBestFit)
    print("RMSE = ",Rmse)
    
    # Enthalpy change of Gaussian fitting
    print(" Enthalpy change of Gaussian fitting 1:", simps(Gaussian(x, popt[0], popt[2], popt[4]), x) )
    print(" Enthalpy change of Gaussian fitting 2:", simps(Gaussian(x, popt[1], popt[3], popt[5]), x) )
               
# Try to use automation to complete multi-peak decomposition        
def AutoMulPeakDecom(x, y):
    
    indexes = peakutils.indexes(y, thres=0.5, min_dist=10)
    #x[indexes], y[indexes]
    Cen = x[indexes]
    Num = len(Cen)*3
    AmpMin = 2
    WidMin = 1
    
    guess = Mul_Param(n = Num, a = AmpMin, m = Cen, s = WidMin)
    popt, pcov = curve_fit(Mul_Gaussian, x, y, guess, maxfev=24000)
    
    print(popt)
    #print(pcov)
    # Gaussian model parameters
    print("[[Variables]]")
    for i in range(len(Cen)): # range(n) gaussian model number
        print("Model Variables", i, ":")
        print("  amp", i, ":", popt[i])
        print("  cen", i, ":", popt[i+1])
        print("  wid", i, ":", popt[i+2])
    
    # Plot peak fitting
    plt.figure(figsize=(8,6))
    plt.plot(x, y, "C2.", label='Net Signal')
    plt.plot(x, Mul_Gaussian(x,*popt), 'C3-', label='Gaussian Fitting')
    #plt.plot(x, Gaussian(x, popt[0], popt[1], popt[2]), 'C4--', label= 'Gausssian 1')
    #plt.plot(x, Gaussian(x, popt[3], popt[4], popt[5]), 'C5--', label= 'Gausssian 2')
    #plt.legend(loc='best', fontsize=16)
    plt.show()              
