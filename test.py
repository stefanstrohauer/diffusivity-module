from PyQt5.QtWidgets import QFileDialog
import json as js
import numpy as np
import pandas as pd
from pathlib import Path, PureWindowsPath
from scipy import io
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.special import erf
from scipy.constants import elementary_charge, Boltzmann
from math import pi
from math import *
import matplotlib as mpl
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages

import os
import timeit
from bisect import bisect_left, bisect_right
import re
import itertools


#T2=DiffusivityMeasurement('./testing_meas/200212_200109A_diffusweep.mat')
#T2=DiffusivityMeasurement('./testing_meas/200212_200109A_diffusweep.mat', 'data')
T2=DiffusivityMeasurement('./testing_meas/200715_200218A_diffusweep-002.json')
T2.RTfit.fit_function_type = 'richards'
# print(T2.fit_function_parameters())
# print(T2.fit_function_parameters(B=0.1))
#print(T2.fit_function_parameters(B='all'))


T2.RTfit.set_fit_parameters(b=0.4)
print(T2.fit_function_parameters(B=0.1))



class Observable:
    def __init__(self):
        self._observers = []

    def register_observer(self, observer):
        self._observers.append(observer)

    def notify_observers(self, *args, **kwargs):
        for observer in self._observers:
            observer.notify(self, *args, **kwargs)

class Observer:
    def __init__(self, observable):
        observable.register_observer(self)

    def notify(self, observable, *args, **kwargs):
        print('Got', args, kwargs, 'From', observable)


subject = Observable()
observer = Observer(subject)
subject.notify_observers('test')


# T2.fit_function_parameters(B=[1,2])
# print(T2.parameters_RTfit)
# T2.set_RT_fit_limits(6,9)
# T2.fit_function_parameters(B=[1,2])
# print(T2.parameters_RTfit)
# T2.set_RT_fit_limits(7,8)
# T2.fit_function_parameters(B=[1,2])
# print(T2.parameters_RTfit)

#T2.get_sheet_resistance(upp_lim=107.18)


# print(T2.fit_function(B=[1.3, 2.4]))
# print(T2.parameters_RTfit)
# print(T2.fitted_RTvalues)
# T2.fit_function_parameters(B='all')
# T2.calc_diffusivity()
# print(T2.get_Dfit_properties())
# print(T2.Bc2_vs_T())
# print(T2.Bc2vsT_fit.linear_fit())

#B=[1,2,3]
#T, R = T2.R_vs_T()


#T2.fit_function_parameters(B='all').values()
#t,r = T2.R_vs_T(B=0.1)
#print(T2.fit_function(t,B=0.1))
#print(T2.unpack_tuple_dictionary(T2.fit_function()))
# R2 = RTfit('richards')
# R2.read_RT_data(T2.R_vs_T(B=0.1))
# R2.fit_data()
# print(R2.Tc())
# R2.return_RTfit(eval_array=10)


#print(R.T, R.R)
# T1,R1,Terr1,Rerr1=T2.R_vs_T(B=np.array([1,2]), err=True) #
# print(T1[1.0])
#a=T.R_vs_B(3, err=True)
#print(a)
