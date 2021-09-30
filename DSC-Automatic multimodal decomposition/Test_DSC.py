import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simps


import BaselineCorrection
from PeakDecomposition import PeakDecom,MulPeakDecom
from AutoMulPeakDecom import AutoMulPeakDecom
#import  MonteCarloPeakDecom_3

#########  first work   ############
#file = r'../combined_dsc1.csv' # BSA protein
#file = r'../calfitter_dsc2.csv' # LinB protein
#file = r'../lysozyme_dsc1.csv' # Lysozyme protein

#########  second work   ############
#file = r'../dvd_dsc1.csv' # dvd protein
#file = r'../dvd_dsc2.csv' # dvd protein, subtract buffer
file = r'../mab_dsc1.csv' # mab protein
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


# Draw
i = 1

# BSA dsc data
#I = I * 687.70764  # Unit conversion, mCal/min ->  KJ/mol/K. BSA DSC Data

# LinB dsc data, from Calfitter web
#I = I

# Lysozyme dsc data
#I = (-I) 

# mab696 dsc data
#I = I + 0.5; # unit conversion, mCal/min -> KJ/mol/K. mab696 DSC Data     

# Net signal, Baseline correction, Probability, Peak Temperature area
Residual, Baseline, ProbClass, PeakSignal, Tm = BaselineCorrection.BaseCorrect(Q, I, fit_num = 4)

'''
# BSA protein, LinB protein
# Unknown baseline estimation, static signal, sample signal probability classification
fig,ax1 = plt.subplots(dpi=600, figsize=(8,6))
plot1=ax1.plot(Q, I[i], c='C0', label='Raw Data')
plot2 = ax1.plot(Q, Baseline[i], c='C1', label='Baseline')
plot3 = ax1.plot(Q, Residual[i], c='C2', label='Net Signal') 
ax2 =ax1.twinx()
plot4=ax2.plot(Q, ProbClass[i], c='C8',label='Distribution probability')

lines = plot1 + plot2 + plot3 + plot4
ax1.legend(lines,[l.get_label() for l in lines],loc='best', fontsize=13)
ax1.set_xlabel("Temperature (℃)", size=22, labelpad=10)
ax1.set_ylabel("Heat capacity (KJ/mol/K)",size=22, labelpad=10)
ax2.set_ylabel("Distribution probability",size=22, labelpad=10)
ax1.tick_params(labelsize=18)
ax2.tick_params(labelsize=18)
plt.show()
'''

'''
# Lysozyme protein
# Unit conversion, w ->  KJ/mol/K. Lysozyme DSC Data
I = (-I) 
Residual, Baseline, ProbClass, PeakSignal, Tm = BaselineCorrection.BaseCorrect(Q, I, fit_num = 4)
times = 17160  # Lysozyme 10mg/mL in the PBS Buffer
#times = 8580 #Lysozyme 20mg/mL in the PBS Buffer
# Unknown baseline estimation, static signal, sample signal probability classification
fig,ax1 = plt.subplots(dpi=600, figsize=(8,6))
plot1=ax1.plot(Q, I[i]*times, c='C0', label='Raw Data')
plot2 = ax1.plot(Q, Baseline[i]*times, c='C1', label='Baseline')
ax2 =ax1.twinx()
plot4=ax2.plot(Q, ProbClass[i], c='C8',label='Distribution probability')

lines = plot1 + plot2  + plot4
ax1.legend(lines,[l.get_label() for l in lines],loc='best', fontsize=14)
ax1.set_xlabel("Temperature (℃)", size=22, labelpad=10)
ax1.set_ylabel("Heat capacity (KJ/mol/K)",size=22, labelpad=10)
ax2.set_ylabel("Distribution probability",size=22, labelpad=10)
ax1.tick_params(labelsize=18)
ax2.tick_params(labelsize=18)
plt.show()
'''

'''
# Mab1 protein, mab696 protein【100/4.186】
# Unknown baseline estimation, static signal, sample signal probability classification
# 1mg/ml:*8880     2 mgl/ml:*4440    5 mgl/ml:*1760
# Kcal/mol/K 的值为 200   218   对于mab696修改之后的应该是 1000/4.186  500/4.186  200/4.186
fig,ax1 = plt.subplots(dpi=600, figsize=(8,6))

#plot1=ax1.plot(Q, I[i] * 200, c='C0', label='Raw Data')
#plot2 = ax1.plot(Q, Baseline[i] * 200, c='C1', label='Baseline')
#plot3 = ax1.plot(Q, Residual[i] * 200, c='C2', label='Net Signal') 
plot1=ax1.plot(Q, I[i]*23 , c='C0', label='Raw Data')
plot2 = ax1.plot(Q, Baseline[i]*23 , c='C1', label='Baseline')
plot3 = ax1.plot(Q, Residual[i]*23 , c='C2', label='Net Signal') 
ax2 =ax1.twinx()
plot4=ax2.plot(Q, ProbClass[i], c='C8',label='Distribution probability')

lines = plot1 + plot2 + plot3 + plot4
ax1.legend(lines,[l.get_label() for l in lines],loc='best', fontsize=13)
ax1.set_xlabel("Temperature (℃)", size=22, labelpad=10)
ax1.set_ylabel("Heat capacity (KJ/mol/K)",size=22, labelpad=10)
ax2.set_ylabel("Distribution probability",size=22, labelpad=10)
ax1.tick_params(labelsize=18)
ax2.tick_params(labelsize=18)
plt.show()
'''


# dvd1 protein, LinB protein
# Unknown baseline estimation, static signal, sample signal probability classification
# 1 mg/ml to convert Kcal/mol/K  *3360    KJ/mol/K *3360/4.18=*803
fig,ax1 = plt.subplots(dpi=600, figsize=(8,6))
# 5 mg/mL KJ/mol/K -> *160
#plot1=ax1.plot(Q, I[i]*160, c='C0', label='Raw Data')
# 10 mg/mL -> KJ/mol/K *80
plot1=ax1.plot(Q, I[i]*80, c='C0', label='Raw Data')
plot2 = ax1.plot(Q, Baseline[i]*80, c='C1', label='Baseline')
plot3 = ax1.plot(Q, Residual[i]*80, c='C2', label='Net Signal') 
ax2 =ax1.twinx()
plot4=ax2.plot(Q, ProbClass[i], c='C8',label='Distribution probability')

lines = plot1 + plot2 + plot3 + plot4
ax1.legend(lines,[l.get_label() for l in lines],loc='best', fontsize=13)
ax1.set_xlabel("Temperature (℃)", size=22, labelpad=10)
ax1.set_ylabel("Heat capacity (KJ/mol/K)",size=22, labelpad=10)
ax2.set_ylabel("Distribution probability",size=22, labelpad=10)
ax1.tick_params(labelsize=18)
ax2.tick_params(labelsize=18)
plt.show()



# Peak decomposition
PeakDecom(Tm[i], PeakSignal[i])
MulPeakDecom(Tm[i], PeakSignal[i])
'''
AutoMulPeakDecom(Tm[i], PeakSignal[i])
'''



