from diffusivity import *

T2=DiffusivityMeasurement('./testing_meas/200212_200109A_diffusweep.mat')
T2.RTfit.fit_function_type = 'richards'
# print(T2.fit_function_parameters())
# print(T2.fit_function_parameters(B=0.1))
#print(T2.fit_function_parameters(B='all'))
T2.fit_function_parameters(B='all')
print(T2.calc_diffusivity(fit_low_lim=7, fit_upp_lim=8))
print(T2.Bc2_vs_T())
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