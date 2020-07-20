from import_lib import *

class DiffusivityMeasurement:
    def __init__(self, filename): # to see why it is cleaner to create new object
                                  # https://stackoverflow.com/questions/29947810/reusing-instances-of-objects-vs-creating-new-ones-with-every-update
        self.filename = filename
        self.key_w_data = 'data'
        self._raw_data = self.__import_file_mat()
        self.time_sweeps = []
        self.T_sweeps = []
        self.B_sweeps = []
        self.R_sweeps = []
        self.__RT_sweeps_per_B = {}
        self.parameters_RTfit = {}
        self.fitted_RTvalues = {}

        self.diffusivity = 0
        self.diffusivity_err = 0
        self.default_B_array = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.B_bin_size = 0.01 #
        self.sheet_resistance_geom_factor = 4.53 # geometry factor to determine sheet resistance 4.53=pi/ln(2)
                                                 # calculated sheet resistance of the film
        self._read_B_sweeps_to_properties()
        self.__rearrange_B_sweeps()
        self.sheet_resistance = self.get_sheet_resistance()
        self.Bc2vsTfit = self.Bc2vsTfit()
        self.RTfit = RTfit()
        self.Tools = Utilities()
        #self.RTfit.read_RT_data(self.__RT_sweeps_per_B[4.0])

    def get_RT(self):
        return self.__RT_sweeps_per_B

    def __import_file_mat(self):
        """reads string with filename and returns .mat data file as list of lists"""
        return io.loadmat(self.filename, squeeze_me=True)[self.key_w_data]

    def _read_B_sweeps_to_properties(self, index_t=8, index_B=9, index_R=10):
        """reads the different measurement quantities from the raw data into class properties"""
        self.time_sweeps = [i[index_t] for i in self._raw_data]
        self.T_sweeps = self.__time_temperature_mapping()
        self.B_sweeps = [i[index_B] for i in self._raw_data]
        self.R_sweeps = [i[index_R] for i in self._raw_data]

    def __rearrange_B_sweeps(self):
        """rearranges the raw data saved in B-sweeps into RT-sweeps"""
        TBR_dict = {'T': self.__flatten_list(self.T_sweeps),
                    'B': self.__flatten_list(self.B_sweeps),
                    'R': self.__flatten_list(self.R_sweeps)}
        TBR_dict['B'] = self.__round_to_base(TBR_dict['B'], base=self.B_bin_size)
        for t, b, r in zip (TBR_dict['T'], TBR_dict['B'], TBR_dict['R']):
            try:
                self.__RT_sweeps_per_B[b].append([t,r])
            except:
                self.__RT_sweeps_per_B[b] = [[t,r]]

        for key, value in self.__RT_sweeps_per_B.items():
            value=np.asarray(value)
            self.__RT_sweeps_per_B[key] = {'T': value[:,0], 'R':value[:,1]}

    def __flatten_list(self, array):
        return np.concatenate(array).ravel()

    def __round_to_base(self, x, base):
      prec = str(base)[::-1].find('.')
      return np.round(base * np.round(x/base),prec)

    def __time_temperature_mapping(self): # TODO: what do we do with this function, maybe change the whole data structure a bit
        Temp_at_timestamp = [i[0:4] for i in self._raw_data]
        T_array = []
        for i,j in zip(self.time_sweeps, Temp_at_timestamp):
            t = [i[0], i[-1]]
            T = [j[2], j[3]]
            p = np.polyfit(t,T,1)
            T_array.append(np.polyval(p,i))
        return T_array

    def R_vs_T(self, B=None, err=False):
        # TODO: rename __array_w_err, __array_dict_w_err, error_std
        # TODO: abbreviation of percent: pc
        B = self.Tools.select_property(B, self.default_B_array, np.array(list(self.__RT_sweeps_per_B.keys())))
        B = self.__round_to_base(B, base=self.B_bin_size)
        if isinstance(B, (int, float)) and not(err):
            return (self.__array_w_err(B, 'T'), self.__array_w_err(B, 'R'))
        elif isinstance(B, (int, float)) and err:
            T_err = self.__array_w_err(B, 'T', self.RTfit.T_meas_error_pp, self.RTfit.T_meas_error_std)
            R_err = self.__array_w_err(B, 'R', self.RTfit.R_meas_error_pp)
            return (self.__array_w_err(B, 'T'), self.__array_w_err(B, 'R'), T_err, R_err)
        elif isinstance(B, (np.ndarray, list)) and not(err):
            return (self.__array_dict_w_err(B, 'T'), self.__array_dict_w_err(B, 'R'))
        elif isinstance(B, (np.ndarray, list)) and err:
            T_dict_err = self.__array_dict_w_err(B, 'T', self.RTfit.T_meas_error_pp, self.RTfit.T_meas_error_std)
            R_dict_err = self.__array_dict_w_err(B, 'R', self.RTfit.R_meas_error_pp)
            return (self.__array_dict_w_err(B, 'T'), self.__array_dict_w_err(B, 'R'), T_dict_err, R_dict_err)
        else: raise TypeError('some input parameter has not a valid type. check input parameters')

    def __array_w_err(self, B, selector, error_pp=1, error_std=0):
        return self.__RT_sweeps_per_B[B][selector] * error_pp + error_std

    def __array_dict_w_err(self, B, selector, error_pp=1, error_std=0):
        return {key:value[selector] * error_pp + error_std for key, value in self.__RT_sweeps_per_B.items() if key in B}

    def get_sheet_resistance(self, upp_lim=450):
        ## TODO: check for correctness of formulas!! and regardgin compatibility to new files -> __flatten_list
        # TODO: while bigger 450 go through sweeps until you find one below
        max_R_NC = np.max(self.__flatten_list(self.R_sweeps))
        if max_R_NC < upp_lim:
            sheet_resistance = self.sheet_resistance_geom_factor * max_R_NC
        else:
            sheet_resistance = self.sheet_resistance_geom_factor * np.max(self.__RT_sweeps_per_B[0.0]['R'])
        return sheet_resistance

    def R_vs_B(self, B_sweep_index, err=False):
        # TODO: abbreviation of percent: pc
        if err:
            B_err = self.B_sweeps[B_sweep_index] * self.Bc2vsTfit.B_field_meas_error_pp + self.Bc2vsTfit.B_field_meas_error_round
            R_err = self.R_sweeps[B_sweep_index] * self.RTfit.R_meas_error_pp
            return (self.B_sweeps[B_sweep_index], self.R_sweeps[B_sweep_index], B_err, R_err)
        else: return (self.B_sweeps[B_sweep_index], self.R_sweeps[B_sweep_index])

    def Bc2_vs_T(self, err=False):
        if not err:
            return (self.Bc2vsT.Bc2vsT['T'], self.Bc2vsT.Bc2vsT['Bc2'])
        elif err:
            return (self.Bc2vsT.Bc2vsT['T'], self.Bc2vsT.Bc2vsT['Bc2'],
                    self.Bc2vsT.Bc2vsT['T_low_err'], self.Bc2vsT.Bc2vsT['T_upp_err'],
                    self.Bc2vsT.Bc2vsT['B_err'])
        else: raise TypeError('"err" function parameter must be boolean type')

    def calc_diffusivity(fit_low_lim=None, fit_upp_lim=None):
        pass

    def fit_function_parameters(self, B=None):
        B = self.Tools.select_property(B, self.default_B_array, np.array(list(self.__RT_sweeps_per_B.keys())))
        if isinstance(B, (int, float)):
            self.RTfit.read_RT_data(self.__RT_sweeps_per_B[B])
            self.parameters_RTfit[B] = self.RTfit.fit_data()
            return self.parameters_RTfit[B]
        elif isinstance(B, (np.ndarray, list)):
            for k in B:
                self.RTfit.read_RT_data(self.__RT_sweeps_per_B[k])
                self.parameters_RTfit[k] = self.RTfit.fit_data()
            return self.parameters_RTfit
        else: raise TypeError('input parameters must have correct type. check input parameters')

    def fit_function(self, T=None, B=None):
        B = self.Tools.select_property(B, self.default_B_array, np.array(list(self.__RT_sweeps_per_B.keys())))
        if isinstance(B, (int, float)):
            self.fit_function_parameters(B)
            return self.RTfit.return_RTfit(*self.set_temperature_array(T), self.parameters_RTfit[B])
        if not set(B).issubset(set(list(self.parameters_RTfit.values()))):
            self.fit_function_parameters(B)
        elif isinstance(T, (dict)) and isinstance(B, (np.ndarray, list)):
            if not set(list(T.keys())).issubset(B):
                raise TypeError("B values are not matching. Please revise B values of temperature dictionary")
            for t, k in zip(T.values(), B):
                self.fitted_RTvalues[k] = self.RTfit.return_RTfit(*self.set_temperature_array(t), self.parameters_RTfit[k])
            return self.fitted_RTvalues
        elif isinstance(B, (np.ndarray, list)):
            for k in B:
                self.fitted_RTvalues[k] = self.RTfit.return_RTfit(*self.set_temperature_array(T), self.parameters_RTfit[k])
            return self.fitted_RTvalues
        else: raise TypeError('input parameters must have correct type. check input parameters')

    def set_temperature_array(self,T):
        request_eval_array = False
        if T is None:
            T_min = np.inf; T_max = 0
            for value in self.__RT_sweeps_per_B.values():
                if np.min(value['T']) < T_min:
                    T_min = np.min(value['T'])
                if np.max(value['T']) > T_max:
                    T_max = np.max(value['T'])
            T_array = np.arange(T_min, T_max, self.RTfit.fit_function_T_default_spacing)
            request_eval_array = True
        elif isinstance(T,(list, np.ndarray)):
            T_array = T
        elif isinstance(T, (int, float)):
            return (T, request_eval_array)
        else: raise TypeError('input parameters must have correct type. check input parameters')
        return (T_array, request_eval_array)

    def unpack_tuple_dictionary(self, dict_a):
        return_dict = {}
        for key, value in dict_a.items():
            return_dict[key] = {'T': value[0], 'R': value[1]}
        return return_dict


    class Bc2vsTfit():
        """docstring for Bc2vsTfit."""

        def __init__(self):
            self.low_lim = None
            self.upp_lim = None
            self.D = 0
            self.dBc2dT = 0
            self.B_0 = 0
            self.err_D = 0
            self.err_dBc2dT = 0
            self.r_squared = 0
            self.Bc2vsT = {}

            self.B_field_meas_error_pp = 0.02 # variation of 1% in voltage monitor results in variation of 1% in Bfield
            self.B_field_meas_error_round = 0.001 # tbreviewed, in Tesla, measurement error + rounding error
            self.linear_fit_T_default_spacing = 0.05



        def linear_fit(arg):
            def linear(self, x, a, b):
                return a*x + b





class RTfit():
    """docstring for RTfit."""

    def __init__(self, fit_function_type = "richards"):
        self.fit_function_type = fit_function_type
        self.fit_function = self.richards
        self.fit_function_T_default_spacing = 0.1
        self.T = 0
        self.R = 0
        self.fit_param = {'output':{}, 'start_values':{}}
        self.curve_fit_options = {}
        self.fit_covariance_matrix = {}

        self.T_meas_error_pp = 0.0125 # in percent, estimated interpolation error
        self.T_meas_error_std = 0.02 # standard deviation of measurements at 4Kelvin
        self.R_meas_error_pp = 0.017 # relative resistance error from resistance measurement
        self.Tools = Utilities()


    def import_data(self, arg):
        pass

    def read_RT_data(self, data):
# # TODO: look if it is necessary to check whether data has something in it!
        if type(data) is dict:
            self.T, self.R = (np.asarray(data['T']), np.asarray(data['R']))
        elif type(data) is np.ndarray:
            shape = np.shape(data)
            if shape[0] is 2:
                self.T, self.R = (np.asarray(data[0,:]), np.asarray(data[1,:]))
            elif shape[1] is 2:
                self.T, self.R = (np.asarray(data[:,0]), np.asarray(data[:,1]))
            else: raise ValueError('array has the shape ' + str(shape))
        elif type(data) is tuple:
            self.T, self.R = data

    def fit_data(self, R_NC=None):
        if self.fit_function_type is "richards":
            self.__define_fitting_parameters_richards(R_NC)
            self.fit_function = self.richards
        elif self.fit_function_type is "gauss_cdf":
            self.__define_fitting_parameters_gauss_cdf(R_NC)
            self.fit_function = self.gauss_cdf
        else: raise ValueError('only "richards" and "gauss_cdf" as possible fitting functions')
        popt, self.fit_covariance_matrix = curve_fit(self.fit_function, self.T, self.R, list(self.fit_param['start_values'].values()), **self.curve_fit_options)
        self.fit_param['output'] = {key: value for key, value in zip(self.fit_param['start_values'].keys(), popt)}
        return self.fit_param['output']


    def __define_fitting_parameters_richards(self, R_NC=None):
        '''Input:
        Output:
        Description: '''
        R_NC = self.Tools.select_property(R_NC, np.max(self.R))
        a,c,q = (0,1,1)
        if self.fit_param['output'] == {}: #and all(abs(start_values_fit_param[list(start_values_fit_param.keys())[1:3]]) < 15)
            k = R_NC #
            nu = 1 # affects near which asymptote maximum growth occurs (nu is always > 0)
            m = a + (k-a)/np.float_power((c+1),(1/nu)) # shift on x-axis
            # TODO: take into account to change b from outside!
            t_2 = self.T[bisect_left(self.R, R_NC/2)]
            b = 1/(m-t_2) * ( np.log( np.float_power((2*(k-a)/k),nu)-c )+np.log(q) ) # growth rate 50
            self.fit_param['start_values'] = {'b': b, 'm': m, 'nu': nu, 'k': k}
        else:
            self.fit_param['start_values'] = self.fit_param['output']
        self.curve_fit_options = {'maxfev': 2500, 'bounds': ([-np.inf, -np.inf, -np.inf, 0.8*R_NC], [np.inf, np.inf, np.inf, 1.2*R_NC])}

    def __define_fitting_parameters_gauss_cdf(self, R_NC=None):
        R_NC = self.Tools.select_property(R_NC, np.max(self.R))
        if bisect_left(self.R, R_NC/2) < len(self.R):
            mean = self.T[bisect_left(self.R, R_NC/2)]
            sigma = self.T[bisect_left(self.R, 0.9*R_NC)]-self.T[bisect_left(self.R, 0.1*R_NC)]
        else:
            R_rev = self.R[::-1]
            mean = self.T[bisect_left(R_rev, R_NC/2)]
            sigma = self.T[bisect_left(R_rev, 0.9*R_NC)]-self.T[bisect_left(R_rev, 0.1*R_NC)]
        if sigma < 0.01:
            sigma = 0.1
        self.fit_param['start_values'] = {'scaling': R_NC, 'mean': mean, 'sigma': sigma}
        self.curve_fit_options = {'maxfev': 1600, 'bounds': (-inf, inf)}

    def richards(self, t,b,m,nu,k, a=0, c=1, q=1):
        return a + (k-a)/np.float_power((c+q*np.exp(-b*(t-m))),1/nu)

    def gauss_cdf(self, x, scaling, mean, sigma):
        return scaling*norm.cdf(x, mean, sigma)

    def return_RTfit(self, eval_array = None, return_eval_array = False, fit_param=None):
        fit_param = self.Tools.select_property(fit_param, self.fit_param['output'])
        if eval_array is None:
            x_low_lim = np.min(self.T)
            x_upp_lim = np.max(self.T)
            eval_array = np.arange(x_low_lim, x_upp_lim, self.fit_function_T_default_spacing)
        if return_eval_array:
            return (eval_array, self.fit_function(eval_array, **fit_param))
        elif not return_eval_array:
            return self.fit_function(eval_array, **fit_param)

    def Tc(self, fit_param = None):
        fit_param = self.Tools.select_property(fit_param, self.fit_param['output'])
        if self.fit_function_type is 'richards':
            return self.__get_Tc_richards(fit_param)
        elif self.fit_function_type is 'gauss_cdf':
            return self.__get_Tc_gauss_cdf()
        else: raise ValueError('only "richards" and "gauss_cdf" as possible fitting functions')

    def __get_Tc_gauss_cdf(self, param):
        if 'mean' in param.keys():
            return param['mean']
        else: raise ValueError('no gaussian parameters found')

    def __get_Tc_richards(self, param):
        if {'b', 'm', 'nu', 'k'}.issubset(list(param.keys())):
            a,c,q = (1,1,1)
            b, m, nu, k = param.values()
            return m - 1/b*( np.log(np.float_power(2*(k-a)/k,nu)-c) + np.log(q) )
        else: raise ValueError('no richards parameters found')


class Utilities():

    def __init__(self):
        pass

    def select_property(self, val, *args):
        if val is None:
            return args[0]
        elif val is 'all':
            return args[1]
        else: return val

T2=DiffusivityMeasurement('./testing_meas/200212_200109A_diffusweep.mat')
T2.RTfit.fit_function_type = 'richards'
# print(T2.fit_function_parameters())
# print(T2.fit_function_parameters(B=0.1))
# print(T2.fit_function_parameters(B=2))
#T2.fit_function_parameters(B='all').values()
t,r = T2.R_vs_T(B=0.1)
print(T2.fit_function(t,B=0.1))
#print(T2.unpack_tuple_dictionary(T2.fit_function()))
# R2 = RTfit()
# R2.read_RT_data(T2.R_vs_T(B=0.1))
# R2.fit_data()
# R2.return_RTfit(eval_array=10)


#print(R.T, R.R)
# T1,R1,Terr1,Rerr1=T2.R_vs_T(B=np.array([1,2]), err=True) #
# print(T1[1.0])
#a=T.R_vs_B(3, err=True)
#print(a)
