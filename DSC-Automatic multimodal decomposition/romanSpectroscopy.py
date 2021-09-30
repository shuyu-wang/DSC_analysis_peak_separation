"""
roman spectroscopy data
"""

import numpy as np
import glob, os
from matplotlib import pyplot
from collections import OrderedDict 

# CZTS roman spectroscopy
path_to_folder = "..\\CZTS_data\\CZTS_111116\\B"
#samples_to_analyse = ['21', '22', '23', '24', '25', '26', '27']
samples_to_analyse = ['21']


def import_data(samples_to_analyse, path_to_folder=path_to_folder):
    dict_container = OrderedDict()
    for sample_id in samples_to_analyse:
        dict_container[sample_id] = []
    # print(dict_container)
   
    for sample_id in samples_to_analyse:
        path = path_to_folder + sample_id + "\\*.txt"
        # print(path)
        for filepath in glob.glob(path):
            # print(filepath)
            data_from_file = np.genfromtxt(filepath)
            dict_container[sample_id].append(data_from_file)
    return dict_container

data = import_data(samples_to_analyse, path_to_folder)

for sample_id, sample_data in data.items():#21, 22 etc
    pyplot.figure( dpi=600)
    pyplot.title("B" + sample_id + " Raman Scattering")
    # print(sample_data)
    
    for idx, data_set in enumerate(sample_data):
        xs = data_set[:,0]
        ys = data_set[:,1]
        data_set[:,1] = 9.5*data_set[:,1]/np.max(data_set[:,1]) ##Make intensity arbitrary units (normalise)
        ys = data_set[:,1]
        pyplot.plot(xs, ys) #Plot individual lines
        
"""
# mass roman spectroscopy test：
path_to_folder_mass = "..\\CZTS_data\\SampleTextFile"

def import_data_mass(samples_to_analyse, path_to_folder):
    dict_container = OrderedDict()
    path = path_to_folder + ".txt"
    # print(path)
    for filepath in glob.glob(path):
        # print(filepath)
        data_from_file = np.genfromtxt(filepath)
        dict_container.append(data_from_file)
    return dict_container

data_mass = import_data_mass(samples_to_analyse, path_to_folder_mass)

for sample_id, sample_data in data_mass.items():#21, 22 etc
    pyplot.figure( dpi=600)
    pyplot.title("SampleTextFile" + " Scattering")
    # print(sample_data)
    
    for idx, data_set in enumerate(sample_data):
        xs = data_set[:,0]
        ys = data_set[:,1]
        #data_set[:,1] = 9.5*data_set[:,1]/np.max(data_set[:,1]) ##Make intensity arbitrary units (normalise)
        ys = data_set[:,1]
        pyplot.plot(xs, ys) #Plot individual lines        
"""        
        
        