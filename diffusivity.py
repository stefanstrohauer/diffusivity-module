from import_lib import *

class DiffusivityMeasurement:
    '''Input: filename with raw data, (fit function type)
    Output: calculation of diffusivity and related values that are stored and/or returned by executed methods
    Description: This class contains all properties concerning the preparation of the measurement data. Also,
    its methods control the workflow to determine the electron diffusivity.'''

    def __init__(self, filename, fit_function='richards'): # to see why it is cleaner to create new object
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
        self.RTfit = RTfit(fit_function)
        self.Bc2vsT_fit = None

    def __import_file_mat(self):
        """Input:filename (from properties)
        Output: list of lists of measurement data
        Description: reads string with filename and returns .mat data file as list of lists"""
        return io.loadmat(self.filename, squeeze_me=True)[self.key_w_data]

    def _read_B_sweeps_to_properties(self, index_t=8, index_B=9, index_R=10):
        """Input: raw measurement data, indices of data lists in raw data
        Output: sets measurement data class properties
        Description: reads the different measurement quantities from the raw data into class properties"""
        self.time_sweeps = [i[index_t] for i in self._raw_data]
        self.T_sweeps = self.__time_temperature_mapping()
        self.B_sweeps = [i[index_B] for i in self._raw_data]
        self.R_sweeps = [i[index_R] for i in self._raw_data]

    def __rearrange_B_sweeps(self):
        """Input: T-sweeps, B-sweeps, R-sweeps
        Output: dictionary {B-field: {T,R}}
        Description: rearranges the raw data saved in B-sweeps into RT-sweeps"""
        TBR_dict = {'T': self.__flatten_list(self.T_sweeps),
                    'B': self.__flatten_list(self.B_sweeps),
                    'R': self.__flatten_list(self.R_sweeps)}
        TBR_dict['B'] = self.__round_to_decimal(TBR_dict['B'], base=self.B_bin_size)
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

    def __round_to_decimal(self, x, base):
      """Input: scalar or array
      Output: scalar or array rounded to arbitrary decimal number, e.g. 0.3
      Description: rounds to arbitrary decimal number. Does so by multiplying the decimal to round with
      the number of times the decimal fits into the scalar/array. In case the scalar/array has less decimals than
      intended to round, a further rounding is performed as outer function."""
      prec = str(base)[::-1].find('.')
      return np.round(base * np.round(x/base), prec)

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
        B = Tools.select_property(B, self.default_B_array, np.array(list(self.__RT_sweeps_per_B.keys())))
        B = self.__round_to_decimal(np.array(B), base=self.B_bin_size)
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
        if self.Bc2vsT_fit is None:
            raise ValueError('to read out data from the Bc2 vs. T relation, execute "calc_diffusivity"')
        if not err:
            return (self.Bc2vsT_fit.Bc2vsT['T'], self.Bc2vsT_fit.Bc2vsT['Bc2'])
        elif err:
            return (self.Bc2vsT_fit.Bc2vsT['T'], self.Bc2vsT.Bc2vsT['Bc2'],
                    self.Bc2vsT_fit.Bc2vsT['T_low_err'], self.Bc2vsT_fit.Bc2vsT['T_upp_err'],
                    self.Bc2vsT_fit.Bc2vsT['B_err'])
        else: raise TypeError('"err" function parameter must be boolean type')

    def calc_diffusivity(self, fit_low_lim=None, fit_upp_lim=None):
        # TODO: split calc diff and print diff (extra method for that)
        B = np.array(list(self.__RT_sweeps_per_B.keys()))
        self.Bc2vsT_fit = self.Bc2vsTfit((*self.get_Tc(B='all', err=True), B), fit_low_lim, fit_upp_lim)
        self.diffusivity, _, _, self.diffusivity_err, _, _ = self.Bc2vsT_fit.calc_Bc2_T_fit()
        return self.Bc2vsT_fit.calc_Bc2_T_fit()

    def get_Tc(self, B=None, err=False):
        B = Tools.select_property(B, self.default_B_array, np.array(list(self.__RT_sweeps_per_B.keys())))
        B = self.__round_to_decimal(np.array(B), base=self.B_bin_size)
        if isinstance(B, (int,float)) and B not in self.parameters_RTfit.keys():
            raise KeyError('no fit parameters found for the asked B field. Please update parameters_RTfit property')
        elif isinstance(B, (list, np.ndarray)) and not set(B).issubset(set(list(self.parameters_RTfit.keys()))):
            raise KeyError('no fit parameters found for the asked B fields. Please update parameters_RTfit property')
        elif not isinstance(B, (int, float, list,np.ndarray)):
            raise KeyError('incorrect type of passed B-fields. Please revise given input parameters of function')
        elif isinstance(B, (int,float)):
            if err is False:
                return self.RTfit.Tc(fit_param=self.parameters_RTfit[B])[0]
            else:
                return self.RTfit.Tc(fit_param=self.parameters_RTfit[B])
        else:
            Tc, Tc_err_low, Tc_err_up = (np.array([]), np.array([]), np.array([]))
            for key in B:
                Tc= np.append(Tc, self.RTfit.Tc(fit_param=self.parameters_RTfit[key])[0])
                Tc_err_low = np.append(Tc_err_low, self.RTfit.Tc(fit_param=self.parameters_RTfit[key])[1])
                Tc_err_up = np.append(Tc_err_up, self.RTfit.Tc(fit_param=self.parameters_RTfit[key])[2])
            if err is False:
                return Tc
            else: return (Tc, Tc_err_low, Tc_err_up)

    def fit_function_parameters(self, B=None):
        B = Tools.select_property(B, self.default_B_array, np.array(list(self.__RT_sweeps_per_B.keys())))
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

    def fit_function(self, B=None, T=None):
        B = Tools.select_property(B, self.default_B_array, np.array(list(self.__RT_sweeps_per_B.keys())))
        if isinstance(B, (int, float)):
            self.fit_function_parameters(B)
            return self.RTfit.return_RTfit(*self.set_temperature_array(T), self.parameters_RTfit[B])
        if not set(B).issubset(set(list(self.parameters_RTfit.keys()))):
            self.fit_function_parameters(B)
        if isinstance(T, (dict)) and isinstance(B, (np.ndarray, list)):
            if not set(list(T.keys())).issubset(B):
                raise TypeError("B values are not matching. Please revise B values of temperature dictionary")
            for t, k in zip(T.values(), B):
                T_data, return_T = self.set_temperature_array(t)
                self.fitted_RTvalues[k] = self.RTfit.return_RTfit(T_data, return_T, self.parameters_RTfit[k])
            return self.fitted_RTvalues
        elif isinstance(B, (np.ndarray, list)):
            for k in B:
                T_data, return_T = self.set_temperature_array(T)
                self.fitted_RTvalues[k] = self.RTfit.return_RTfit(T_data, return_T, self.parameters_RTfit[k])
        else: raise TypeError('input parameters must have correct type. check input parameters')
        print(return_T)
        if return_T:
            T_fit, R_fit = ({}, {})
            for key, value in self.fitted_RTvalues.items():
                T_fit[key], R_fit[key] = (value[0], value[1])
            return (T_fit, R_fit)
        else: return self.fitted_RTvalues

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
        elif isinstance(T, (list, np.ndarray)):
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

        def __init__(self, data, fit_low_lim, fit_upp_lim):
            self.low_lim = None
            self.upp_lim = None
            self.__D = 0
            self.__dBc2dT = 0
            self.__B_0 = 0
            self.__err_D = 0
            self.__err_dBc2dT = 0
            self.__r_squared = 0
            self.Bc2vsT = {}
            self.Tc = 0

            self.B_field_meas_error_pp = 0.02 # variation of 1% in voltage monitor results in variation of 1% in Bfield
            self.B_field_meas_error_round = 0.001 # tbreviewed, in Tesla, measurement error + rounding error
            self.linear_fit_T_default_spacing = 0.05

            self.set_properties(data)
            self.low_lim = Tools.select_property(fit_low_lim, np.sort(self.Bc2vsT['T'])[1])
            self.upp_lim = Tools.select_property(fit_upp_lim, np.sort(self.Bc2vsT['T'])[-2])

        def set_properties(self, data):
            self.Bc2vsT['T'], self.Bc2vsT['T_low_err'], self.Bc2vsT['T_upp_err'], self.Bc2vsT['Bc2'] = data

        def linear_fit(self, T=None):
            T_min_def = np.min(self.Bc2vsT['T'])
            T_max_def = np.max(self.Bc2vsT['T'])
            if T is None:
                T = Tools.select_property(T, np.arange(T_min_def, T_max_def, self.linear_fit_T_default_spacing))
                return (T, self.__dBc2dT*T + self.__B_0)
            elif isinstance(T, (list, np.ndarray)):
                return self.__dBc2dT*T + self.__B_0
            else:
                raise TypeError('wrong type of input parameter T. Please check input parameters')

        def calc_Bc2_T_fit(self, fit_low_lim=None, fit_up_lim=None):
            T_fit_array, Bc2_fit_array = self.__select_T_Bc2(fit_low_lim, fit_up_lim)
            # print(self.Bc2vsT['T'])
            # print(T_fit_array)
            # print('\n')
            # print(self.Bc2vsT['Bc2'])
            # print(Bc2_fit_array)
            # print(len(self.Bc2vsT['T']), len(T_fit_array))
            # print(len(self.Bc2vsT['Bc2']), len(Bc2_fit_array))
            self.__dBc2dT, self.__B_0, r_value, _, self.__err_dBc2dT = \
                linregress(T_fit_array, Bc2_fit_array)
            self.__r_squared = r_value**2
            self.__D = -4*Boltzmann/(pi*elementary_charge*self.__dBc2dT)*1e4
            self.__err_D = abs(4*Boltzmann/(pi*elementary_charge*(self.__dBc2dT**2))*self.__err_dBc2dT)
            self.Tc = -self.__B_0/self.__dBc2dT
            return self.__get_fit_properties()

        def __select_T_Bc2(self, fit_low_lim=None, fit_up_lim=None):
            fit_low_lim = Tools.select_property(fit_low_lim, self.low_lim)
            fit_up_lim = Tools.select_property(fit_up_lim, self.upp_lim)
            T_Bc2 = np.array([self.Bc2vsT['T'], self.Bc2vsT['Bc2']]).transpose()
            T_Bc2 = T_Bc2[T_Bc2[:,0].argsort()]
            #TODO: check for a better condition where only one line is used
            T_Bc2 = T_Bc2[T_Bc2[:,0] > fit_low_lim, :]
            T_Bc2 = T_Bc2[T_Bc2[:,0] < fit_up_lim, :]
            T_array, Bc2_array = (T_Bc2[:,0], T_Bc2[:,1])
            return (T_array, Bc2_array)

        def __get_fit_properties(self):
            return (self.__D, self.__dBc2dT, self.__B_0, self.__err_D, self.__err_dBc2dT, self.__r_squared)


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

    def read_RT_data(self, data):
        self.T, self.R = Tools.read_data_to_properties(data)

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
        R_NC = Tools.select_property(R_NC, np.max(self.R))
        a,c,q = (0,1,1)
        if self.fit_param['output'] == {}: #and all(abs(start_values_fit_param[list(start_values_fit_param.keys())[1:3]]) < 15)
            k = R_NC  #
            nu = 1  # affects near which asymptote maximum growth occurs (nu is always > 0)
            m = a + (k-a)/np.float_power((c+1),(1/nu))  # shift on x-axis
            # TODO: take into account to change b from outside!
            t_2 = self.T[bisect_left(self.R, R_NC/2)]
            b = 1/(m-t_2) * ( np.log( np.float_power((2*(k-a)/k),nu)-c )+np.log(q) ) # growth rate 50
            self.fit_param['start_values'] = {'b': b, 'm': m, 'nu': nu, 'k': k}
        else:
            self.fit_param['start_values'] = self.fit_param['output']
        self.curve_fit_options = {'maxfev': 2500, 'bounds': ([-np.inf, -np.inf, -np.inf, 0.8*R_NC], [np.inf, np.inf, np.inf, 1.2*R_NC])}

    def __define_fitting_parameters_gauss_cdf(self, R_NC=None):
        R_NC = Tools.select_property(R_NC, np.max(self.R))
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
        fit_param = Tools.select_property(fit_param, self.fit_param['output'])
        if eval_array is None:
            x_low_lim = np.min(self.T)
            x_upp_lim = np.max(self.T)
            eval_array = np.arange(x_low_lim, x_upp_lim, self.fit_function_T_default_spacing)
        if return_eval_array:
            return (eval_array, self.fit_function(eval_array, **fit_param))
        elif not return_eval_array:
            return self.fit_function(eval_array, **fit_param)

    def Tc(self, fit_param = None):
        fit_param = Tools.select_property(fit_param, self.fit_param['output'])
        if self.fit_function_type is 'richards':
            Tc = self.__get_Tc_richards(fit_param)
            return (Tc, *self.__Tc_error(Tc))
        elif self.fit_function_type is 'gauss_cdf':
            Tc = self.__get_Tc_gauss_cdf(fit_param)
            return (Tc, *self.__Tc_error(Tc))
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
        else:
            raise ValueError('no richards parameters found')

    def __Tc_error(self, Tc):
        T_data_from_below = self.T[bisect_left(self.T, Tc) - 1]
        T_data_from_above = self.T[bisect_left(self.T, Tc)]
        T_err_low = abs(Tc - T_data_from_below - self.T_meas_error_pp * T_data_from_below - self.T_meas_error_std)
        T_err_up = abs(T_data_from_above + self.T_meas_error_pp * T_data_from_above + self.T_meas_error_std - Tc)
        return (T_err_low, T_err_up)


class Tools():

    @staticmethod
    def select_property(val, *args):
        if val is None:
            return args[0]
        elif val is 'all':
            return args[1]
        else: return val

    @staticmethod
    def read_data_to_properties(data):
    # TODO: look if it is necessary to check whether data has something in it!
    # TODO: Check to make this function more useful and not only for RT fit class
        if type(data) is dict:
            return (np.asarray(data['T']), np.asarray(data['R']))
        elif type(data) is np.ndarray:
            shape = np.shape(data)
            if shape[0] is 2:
                return (np.asarray(data[0,:]), np.asarray(data[1,:]))
            elif shape[1] is 2:
                return (np.asarray(data[:,0]), np.asarray(data[:,1]))
            else: raise ValueError('array has the shape ' + str(shape))
        elif type(data) is tuple:
            return data
