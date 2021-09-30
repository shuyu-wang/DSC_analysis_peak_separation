import numpy as np
from copy import deepcopy


### Iterative polynomial fitting generate baseline ###
def PolyFit(fit_x, fit_y, fit_num = 5,pt = 1):
    
    fit_x = np.array(fit_x)
    fit_y = np.array(fit_y)
    fit_num = fit_num 
    f1 = np.polyfit(fit_x, fit_y, fit_num)  
            
    # Use np.poly1d() function to solve the fitted curve
    b0 = np.poly1d(f1) # b0 is the power of polynomial fitting
    b1 = list( map(b0, fit_x) )
    b = np.array(b1)
            
    pt = 1
    y0 = deepcopy(fit_y)
    bk = deepcopy(fit_y)
    #while pt > 0.001:
    while pt > 0.0001:
    #while pt > 0.0004:
        for i in range(0,len(fit_y)-1):
            if y0[i] > b[i]:
                y0[i] = b[i]
        z = y0 -bk
        z0 = 0
        bn = 0
        for i in range(0,len(fit_y)-1):
            z0 = z0 + z[i]**2
            bn = bn + bk[i]**2
        pt = np.sqrt(z0)/np.sqrt(bn)
        f1 = np.polyfit(fit_x, y0,fit_num)
        b0 = np.poly1d(f1)
        b1 = list(map(b0, fit_x))
        b = np.array(b1)
        bk = deepcopy(y0)
    
    return y0


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    #file = r'../combined_dsc1.csv' # BSA dsc data
    #file = r'../calfitter_dsc2.csv'# LinB dsc data, Calfitter
    #file = r'../lysozyme_dsc1.csv' # lysozyme dsc data
    
    
    #file = r'../dvd_dsc1.csv' # dvd protein
    #file = r'../dvd_dsc2.csv' # dvd protein, subtract buffer
    #file = r'../mab_dsc1.csv' # mab protein
    file = r'../mab696_dsc1.csv'
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

    i = 0
    x = I[i] 
    Baseline = PolyFit(fit_x = Q, fit_y = x, fit_num = 4,pt = 1)
    NetSignal = x - Baseline 
    
    plt.figure(dpi=600, figsize=(8,6))
    plt.plot(Q, x, label='Raw Data')
    plt.plot(Q, PolyFit(Q, x), label='Poly Baseline')
    plt.legend()
    plt.show()
    
    plt.plot(Q, NetSignal, label='Net Data')
    plt.legend()
    plt.show()
    
 