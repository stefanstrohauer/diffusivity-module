####################################
# Module for evaluation of diffusivity measurements
# Authors: Stefan Strohauer, Noah Ploch
# 2019 - 2020
####################################

## R_NC muss noch in
# maybe check meaningfulness of error calculation of T_Bc2
# check for "is false"

from import_lib import *
#import json as js

class DiffusivityMeasurement:
    '''Input: filename with raw data, (fit function type)
    Output: calculation of diffusivity and related values that are stored and/or returned by executed methods
    Description: This class contains all properties concerning the preparation of the measurement data. Also,
    its methods control the workflow to determine the electron diffusivity.'''

    def __init__(self, filename, key_w_data=None, fit_function='richards', T_sweeps=None, T_selector = "T_sample"): # to see why it is cleaner to create new object
                                  # https://stackoverflow.com/questions/29947810/reusing-instances-of-objects-vs-creating-new-ones-with-every-update
        self.filename = str(filename)
        self.key_w_data = key_w_data  # for old measurement schemes: data, data_old

        # here measurement data is stored in a structured manner
        self.time_sweeps = []
        self.T_sample_sweeps = []
        self.T_PCB_sweeps = []
        self.T_sample_ohms_sw = []
        self.T_PCB_ohms_sw = []
        self.B_sweeps = []
        self.R_sweeps = []
        self._raw_data = self.__import_file(key_w_data)

        self.__RT_sweeps_per_B = {}  # sorting temperature and resistance after magnetic field (B: {T,R})
        self.parameters_RTfit = {}  # dictionary of B fields with corresponding fit parameters from fitting {T,R}   {B:{fit parameters}}
        self.fitted_RTvalues = {}  # fitted values of Resistance R for given Temperature T at magnetic fields B
        self.__RT_fit_low_lim = 0  # lower limit of RT fits
        self.__RT_fit_upp_lim = np.inf  # upper limit of RT fits

        self.diffusivity = 0
        self.diffusivity_err = 0
        self.Tc_0T = 0
        self.T_selector = T_selector # options: T_sample or T_PCB
        self.default_B_array = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.B_bin_size = 0.01 #
        self.sheet_resistance_geom_factor = 4.53 # geometry factor to determine sheet resistance 4.53=pi/ln(2)
                                                 # calculated sheet resistance of the film
        self.__rearrange_B_sweeps(T_sweeps)
        self.max_measured_resistance = self.get_R_NC()
        self.sheet_resistance = self.sheet_resistance_geom_factor*self.max_measured_resistance

        self.RTfit = RTfit(fit_function)  # initializing RT fit class
        self.Bc2vsT_fit = None

    def __import_file(self, key_with_data=None):
        """Input: data key with measurement data, filename (from properties)
        Output: structured measurement data
        Description: reads string with filename and returns measurement data from json and .mat files. Also, reads into the
        corresponding attributes the measurement data. Differentiates between json and mat file through setting the key_with_data parameter"""
        if set('.json').issubset(set(self.filename)):
            with open(self.filename) as json_data:
                data_raw = js.load(json_data)
                self.time_sweeps, self.T_sample_sweeps, self.T_PCB_sweeps, self.T_sample_ohms_sw, self.T_PCB_ohms_sw, self.B_sweeps, self.R_sweeps = \
                    self.__return_measurement_data(data_raw['sweep'], 't', 'T_sample', 'T_PCB', 'T_sample_ohms', 'T_PCB_ohms', 'B', 'R')
        elif set('.mat').issubset(set(self.filename)):
            data = io.loadmat(self.filename, squeeze_me=True)
            if key_with_data in list(data.keys()):  # this try except structure tries to look up the data through the given dict key and, if it is None or not valid (none is not valid), tries the dict keys we usually used in the past measurements.
                data_raw = data[key_with_data]
            elif ('data' in data.keys()) or ('data_old' in data.keys()):
                key_list = list(data.keys())
                self.key_w_data = list(filter(lambda x:'data' in x, key_list))[0] # filters from the keys in the data dictionary if it is data or data_old
                data_raw = data[self.key_w_data]
            else:
                raise Exception('Unknown measurement. Data was not found in .mat structure')
            self.time_sweeps, self.B_sweeps, self.R_sweeps = self.__return_measurement_data(data_raw, 8, 9, 10)
            self.T_sample_sweeps = self.__time_temperature_mapping(data_raw)
        else: raise Exception('wrong data type: Please check if correct parameters were passed')
        return data_raw

    def __return_measurement_data(self, data, *args):
        """Input: raw measurement data, indices/keys of measurment quantities dict in raw data
        Output: list of measurement quantites to be unpacked
        Description: reads the different measurement quantities from the raw data into a list that can be unpacked""" #index_t=8, index_B=9, index_R=10
        data_to_properties = []
        for arg in args:
            data_to_properties.append([i[arg] for i in data])
        return data_to_properties

    def __rearrange_B_sweeps(self, T_values = None):
        """Input: T-sweeps, B-sweeps, R-sweeps
        Output: dictionary {B-field: {T,R}}
        Description: rearranges the raw data saved in B-sweeps into RT-sweeps"""
        if self.T_selector == "T_sample":
            T_sweeps = Tools.selector(T_values, self.T_sample_sweeps)
        elif self.T_selector == "T_PCB":
            T_sweeps = Tools.selector(T_values, self.T_PCB_sweeps)
        elif T_values != None:  # maybe these 2 lines can be deleted as this
            T_sweeps = T_values # case is already coverd above with Tools.selector
        else:
            raise ValueError('Only T_sample and T_PCB are valid options for T_selector.')

        TBR_dict = {'T': self.__flatten_list(T_sweeps),
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

    def __time_temperature_mapping(self, data):
        '''Input: temperature data
        Output: interpolated temperature data with same length as R and B-sweeps
        Description: For measurements without Cernox. As temperature is measured between B-sweeps, the time array is used
        to interpolate the temperature (temperature is assumed to rise almost linearly).'''
        # the old data structure is: data =
        # prev_temp_diode, curr_temp_diode, prev_temp_sample, curr_temp_sample,
        # prev_volt_diode, curr_volt_diode, prev_volt_sample, curr_volt_sample,
        # curr_time, curr_mag_field, curr_resistance
        Temp_at_timestamp = [i[0:4] for i in data]
        T_array = []
        for i,j in zip(self.time_sweeps, Temp_at_timestamp):
            t = [i[0], i[-1]]  # locations of the time and temperature data
            T = [j[2], j[3]] # indices 2, 3 yield temperature of the sample diode
            p = np.polyfit(t,T,1)
            T_array.append(np.polyval(p,i))
        return T_array

    def R_vs_T(self, B=None, err=False):
        '''Input: B fields for which RvsT should be returned, flag to decide if error is returned
        Output: Tuple of varying size depending on err-flag including (T,R,T_err,R_err) of an RT-sweep (Resistance-Temperature-Sweep)
        Description: Depending on the input B fields "B" and the err-flag, this method returns the associated RT-sweep with error'''
        B = Tools.selector(B, self.default_B_array, np.array(list(self.__RT_sweeps_per_B.keys())))
        B = self.__round_to_decimal(np.array(B), base=self.B_bin_size)
        if isinstance(B, (int, float)) and not(err):
            return (self.__RT_array_builder(B, 'T'), self.__RT_array_builder(B, 'R'))
        elif isinstance(B, (int, float)) and err:
            T_err = self.__RT_array_builder(B, 'T', self.RTfit.T_meas_error_pc, self.RTfit.T_meas_error_std_dev)
            R_err = self.__RT_array_builder(B, 'R', self.RTfit.R_meas_error_pc)
            return (self.__RT_array_builder(B, 'T'), self.__RT_array_builder(B, 'R'), T_err, R_err)
        elif isinstance(B, (np.ndarray, list)) and not(err):
            return (self.__RT_dict_builder(B, 'T'), self.__RT_dict_builder(B, 'R'))
        elif isinstance(B, (np.ndarray, list)) and err:
            T_dict_err = self.__RT_dict_builder(B, 'T', self.RTfit.T_meas_error_pc, self.RTfit.T_meas_error_std_dev)
            R_dict_err = self.__RT_dict_builder(B, 'R', self.RTfit.R_meas_error_pc)
            return (self.__RT_dict_builder(B, 'T'), self.__RT_dict_builder(B, 'R'), T_dict_err, R_dict_err)
        else: raise TypeError('some input parameter has not a valid type. check input parameters')

    def __RT_array_builder(self, B, selector, error_pc=1, error_std=0):  # Method to return either the R (T) array or the R_err (T_err) array
        return self.__RT_sweeps_per_B[B][selector] * error_pc + error_std

    def __RT_dict_builder(self, B, selector, error_pc=1, error_std=0):  # Method returning either R(R_err)/T(T_err) arrays in form of dicts (B as key)
        return {key:value[selector] * error_pc + error_std for key, value in self.__RT_sweeps_per_B.items() if key in B}

    def get_R_NC(self, upp_lim=np.Inf):
        '''Input: upper limit for measured resistance
        Output: maximum measured resistance (of all sweeps) within limits
        Description: Determine maximum measured resistance (usually close to 15K). If value is not reasonable,
        every sweep is searched for its maximum until one is found with R_meas_max < upp_lim'''
        max_R_NC = 0
        for value in self.__RT_sweeps_per_B.values():
            curr_R = np.max(value['R'])
            if curr_R > max_R_NC and curr_R <= upp_lim:
                 max_R_NC = curr_R

        self.max_measured_resistance = max_R_NC
        self.sheet_resistance = max_R_NC*self.sheet_resistance_geom_factor
        return max_R_NC

    def R_vs_B(self, B_sweep_index, err=False):
        '''Input: index of the searched B sweep in the data
        Output: R-B-sweep as measured, optionally with error
        Description: Returns R-B-sweeps'''
        if err:
            B_err = self.B_sweeps[B_sweep_index] * self.Bc2vsT_fit.B_field_meas_error_pc + self.Bc2vsT_fit.B_field_meas_error_round
            R_err = self.R_sweeps[B_sweep_index] * self.RTfit.R_meas_error_pc
            return (self.B_sweeps[B_sweep_index], self.R_sweeps[B_sweep_index], B_err, R_err)
        else: return (self.B_sweeps[B_sweep_index], self.R_sweeps[B_sweep_index])

    def Bc2_vs_T(self, err=False):
        '''Input: error flag
        Output: Bc2(T) as arrays (T,Bc2), (optional) with error
        Description: returns Bc2(T). Data comes from the Bc2vsTfit class initialied in calc_diffusivity'''
        if self.Bc2vsT_fit is None:
            raise ValueError('to read out data from the Bc2 vs. T relation, execute "calc_diffusivity"')
        if not err:
            return (self.Bc2vsT_fit.Bc2vsT['T'], self.Bc2vsT_fit.Bc2vsT['Bc2'])
        elif err:
            return (self.Bc2vsT_fit.Bc2vsT['T'], self.Bc2vsT_fit.Bc2vsT['Bc2'],
                    self.Bc2vsT_fit.Bc2vsT['T_low_err'], self.Bc2vsT_fit.Bc2vsT['T_upp_err'],
                    self.Bc2vsT_fit.Bc2vsT['Bc2_err'])
        else: raise TypeError('"err" function parameter must be boolean type')

    def calc_diffusivity(self, fit_low_lim=None, fit_upp_lim=None):
        '''Input: fit limits of the Bc2(T)
        Output: Diffusivity and related values are set in the properties
        Description: First all B fields (keys) are selected. Then the Bc2vsTfit class is initialized and the diffusivity and related
        values calculated and the respective properties set'''
        B = np.array(list(self.__RT_sweeps_per_B.keys()))
        self.Bc2vsT_fit = self.Bc2vsTfit((*self.get_Tc(B='all', err=True), B), fit_low_lim, fit_upp_lim)  # initiaizing class
        self.Bc2vsT_fit.calc_Bc2_T_fit()  # calculating diffusivity and values
        self.diffusivity, _, _, self.diffusivity_err, _, _ = self.Bc2vsT_fit.get_fit_properties()  # setting properties
        self.Tc_0T = self.Bc2vsT_fit.Tc  # handig over Tc (calculated in Bc2vsTfit)
        return self.Bc2vsT_fit.get_fit_properties()

    def get_Bc2vsT_fit(self, T=None):
        return self.Bc2vsT_fit.linear_fit(T)

    def get_Tc(self, B=None, err=False):
        '''Input: B fields, flag if error should be returned
        Output: Transition temperature Tc of RT-sweep (with lower und upper error)
        Description: Sets B array and checks if corresponding RT sweeps have been fitted. Depending on type of B, returns tuple or tuple of arrays
        with Tc values'''
        B = Tools.selector(B, self.default_B_array, np.array(list(self.__RT_sweeps_per_B.keys())))
        B = self.__round_to_decimal(np.array(B), base=self.B_bin_size)
        self.__checkif_B_in_param(B)
        R_NC = self.max_measured_resistance
        if isinstance(B, (int,float)):
            if err is False:
                return self.RTfit.Tc(fit_param=self.parameters_RTfit[B], R_NC=R_NC)[0]
            else:
                return self.RTfit.Tc(fit_param=self.parameters_RTfit[B], R_NC=R_NC)
        else:
            Tc, Tc_err_low, Tc_err_up = (np.array([]), np.array([]), np.array([]))
            for key in B:
                Tc= np.append(Tc, self.RTfit.Tc(fit_param=self.parameters_RTfit[key], R_NC=R_NC)[0])
                Tc_err_low = np.append(Tc_err_low, self.RTfit.Tc(fit_param=self.parameters_RTfit[key], R_NC=R_NC)[1])
                Tc_err_up = np.append(Tc_err_up, self.RTfit.Tc(fit_param=self.parameters_RTfit[key], R_NC=R_NC)[2])
            if err is False:
                return Tc
            else: return (Tc, Tc_err_low, Tc_err_up)

    def fit_function_parameters(self, B=None):
        '''Input: B fields as array or scalar
        Output: fitting parameters for chosen B fields
        Description: Sets chosen B fields and returns dictionary of fit parameters with B fields as keys'''
        B = Tools.selector(B, self.default_B_array, np.array(list(self.__RT_sweeps_per_B.keys())))
        B = self.__round_to_decimal(np.array(B), base=self.B_bin_size)  # This and above line needed to assure B values are valid and rounded to correct decimal
        if isinstance(B, (int, float)):
            self.RTfit.read_RT_data(self.__RT_sweeps_per_B[B])
            self.parameters_RTfit[B] = self.RTfit.fit_data(fit_low_lim=self.__RT_fit_low_lim, fit_upp_lim=self.__RT_fit_upp_lim)
            return self.parameters_RTfit[B]
        # Difference between scalar and array to be able to return dict or dict of dicts
        elif isinstance(B, (np.ndarray, list)):
            return_fit_param_dict = {}
            for k in B:
                self.RTfit.read_RT_data(self.__RT_sweeps_per_B[k])
                self.parameters_RTfit[k] = self.RTfit.fit_data(fit_low_lim=self.__RT_fit_low_lim, fit_upp_lim=self.__RT_fit_upp_lim)
                return_fit_param_dict[k] = self.parameters_RTfit[k].copy()
            return return_fit_param_dict
        else: raise TypeError('input parameters must have correct type. check input parameters')

    def calc_RT_fits(self):
        self.fit_function_parameters(B='all')

    # def set_RT_fit_limits(self, fit_low_lim, fit_upp_lim):  # sets upper and lower limit for the fits of RT sweeps
    #     self.__RT_fit_low_lim = fit_low_lim
    #     self.__RT_fit_upp_lim = fit_upp_lim

    def get_RT_fit_limits(self): # gets upper and lower limit for the fits of RT sweeps
        return (self.__RT_fit_low_lim, self.__RT_fit_upp_lim)

    def fit_function(self, B=None, T=None):
        '''Input: B field as array or scalar, Temperature T as array, dict or scalar
        Output: tuple of arrays, scalars or dicts of the fitted RT values
        Description: Sets B to valid values and checks if fit parameters exist. Varying tuple entries depending on input of B and T.
        For T=None, the T array is returned as well. Possible to hand over personalized T arrays or the T arrays of the measurement corresponding to the B fields entered'''
        B = Tools.selector(B, self.default_B_array, np.array(list(self.__RT_sweeps_per_B.keys())))
        B = self.__round_to_decimal(np.array(B), base=self.B_bin_size)
        self.__checkif_B_in_param(B)
        if isinstance(B, (int, float)):
            return self.RTfit.return_RTfit(*self.set_temperature_array(T), self.parameters_RTfit[B])
        elif isinstance(T, (dict)) and isinstance(B, (np.ndarray, list)):
            if not set(list(T.keys())).issubset(B):  # used to check if B fields of T array (keys) are included in the given B values
                raise TypeError("B values are not matching. Please revise B values of temperature dictionary")
            for t, k in zip(T.values(), B):
                T_data, return_T = self.set_temperature_array(t)
                self.fitted_RTvalues[k] = self.RTfit.return_RTfit(T_data, return_T, self.parameters_RTfit[k])
        else:
            for k in B:
                T_data, return_T = self.set_temperature_array(T)
                self.fitted_RTvalues[k] = self.RTfit.return_RTfit(T_data, return_T, self.parameters_RTfit[k])
            self.fitted_RTvalues = self.__unpack_tuple_dictionary(self.fitted_RTvalues)
        if return_T:  # self.fitted_RTvalues is a dictionary with tuples as values. To keep systematic, a tuple of dicts must be returned, as happend here
            T_fit, R_fit = ({}, {})
            for key, value in self.fitted_RTvalues.items():
                T_fit[key], R_fit[key] = (value['T'], value['R'])
            return (T_fit, R_fit)
        else: return self.fitted_RTvalues

    def set_temperature_array(self,T):
        '''Input: T array
        Output: Adapted T array
        Description: Sets T array correctly according to specifications: if None: maximal T range of measurement, else return T. Also, this method sets
        if T array should be returned as well in fit_function'''
        request_eval_array = False
        if T is None:
            T_min = np.inf; T_max = 0
            for value in self.__RT_sweeps_per_B.values():  # looking for max (min) values
                if np.min(value['T']) < T_min:
                    T_min = np.min(value['T'])
                if np.max(value['T']) > T_max:
                    T_max = np.max(value['T'])
            T_array = np.linspace(T_min, T_max, num=self.RTfit.fit_function_number_of_datapoints)
            request_eval_array = True
        elif isinstance(T, (list, np.ndarray)):
            T_array = T
        elif isinstance(T, (int, float)):
            return (T, request_eval_array)
        else: raise TypeError('input parameters must have correct type. check input parameters')
        return (T_array, request_eval_array)

    def __checkif_B_in_param(self, B):
        '''Input: B array
        Output: -
        Description: Checks whether B fields in B array have already been fitted, otherwise calls fit_function_parameters'''
        not_in_param_scalar = isinstance(B, (int,float)) and B not in self.parameters_RTfit.keys()
        not_in_param_array = isinstance(B, (list, np.ndarray)) and not set(B).issubset(set(list(self.parameters_RTfit.keys())))
        if not_in_param_scalar or not_in_param_array:
            self.fit_function_parameters(B)
        elif not isinstance(B, (int, float, list,np.ndarray)):
            raise TypeError('input parameters must have correct type. check input parameters')

    def __unpack_tuple_dictionary(self, dict_input):  # returns dictionary with keys for T and R taken from the dictionary of tuples
        return_dict = {}
        for key, value in dict_input.items():
            return_dict[key] = {'T': value[0], 'R': value[1]}
        return return_dict


    class Bc2vsTfit():
        """This class object contains the fit of the Bc2-T relation, performing all necessary steps to calculate up to the diffusivity.
        It takes as input the Bc2 and T arrays determined from the RT fitting. These values are fitted with linear regression and the
        diffusivity and related values are calculated and stored within the class object. It is a nested class from DiffusivityMeasurement."""

        def __init__(self, data, fit_low_lim, fit_upp_lim):
            self.__D = 0
            self.__dBc2dT = 0
            self.__B_0 = 0
            self.__err_D = 0
            self.__err_dBc2dT = 0
            self.__r_squared = 0
            self.Bc2vsT = {}  # here the (T,Bc2) data points determined from the measurement and fitting are stored
            self.Tc = 0

            self.B_field_meas_error_pc = 0.02 # variation of 2% in voltage monitor results in variation of 1% in Bfield
            self.B_field_meas_error_round = 0.001 # tbreviewed, in Tesla, measurement error + rounding error
            self.linear_fit_number_of_datapoints = 100  # amount of data points to be placed between lower and upper T values for the linear fit

            self.set_properties(data)  # read data into attributes
            self.low_lim = Tools.selector(fit_low_lim, np.sort(self.Bc2vsT['T'])[1])  # determine lower fit limit, if None the minimum of T is taken
            self.upp_lim = Tools.selector(fit_upp_lim, np.sort(self.Bc2vsT['T'])[-2]) # determine upper fit limit, if None the maximum of T is taken

        def set_properties(self, data): # set the attributes of the class feeding them with data provided from the outer class
            self.Bc2vsT['T'], self.Bc2vsT['T_low_err'], self.Bc2vsT['T_upp_err'], self.Bc2vsT['Bc2'] = data
            self.Bc2vsT['Bc2_err'] = self.Bc2vsT['Bc2']*self.B_field_meas_error_pc + self.B_field_meas_error_round

        def linear_fit(self, T=None):
            '''Input: Temperature values to be evaluated for Bc2
            Output: Bc2-array or (T,Bc2) tuple
            Description: takes the T array and, using the slope and the intercept calculated, evalates T values.'''
            T_min_def = np.min(self.Bc2vsT['T'])
            T_max_def = np.max(self.Bc2vsT['T'])
            if T is None:
                T = Tools.selector(T, np.linspace(T_min_def, T_max_def, num=self.linear_fit_number_of_datapoints))
                return (T, self.__dBc2dT*T + self.__B_0)
            elif isinstance(T, (list, np.ndarray)):
                return self.__dBc2dT*T + self.__B_0
            else:
                raise TypeError('wrong type of input parameter T. Please check input parameters')

        def calc_Bc2_T_fit(self):
            '''Input: uses attributes from the class object
            Output: sets attributes of the class object
            Description: Defines T-Bc2 values to be fitted through lower and upper fit limit. Calculates the fit with linear regression.
            Calculates diffusivity, diffusivity error and Tc(0T).'''
            T_fit_array, Bc2_fit_array = Tools.select_values(self.Bc2vsT['T'], self.Bc2vsT['Bc2'], self.low_lim, self.upp_lim)
            if T_fit_array.size == 0:
                raise ValueError('chosen fit limits result in empty array. please change fit limits')
            self.__dBc2dT, self.__B_0, r_value, _, self.__err_dBc2dT = \
                linregress(T_fit_array, Bc2_fit_array)
            self.__r_squared = r_value**2
            self.__D = -4*Boltzmann/(pi*elementary_charge*self.__dBc2dT)*1e4
            self.__err_D = abs(4*Boltzmann/(pi*elementary_charge*(self.__dBc2dT**2))*self.__err_dBc2dT)  # from gaussian error propagation
            self.Tc = -self.__B_0/self.__dBc2dT

        def get_fit_properties(self):  # returns the fit properties/physical properties since they are private
            return (self.__D, self.__dBc2dT, self.__B_0, self.__err_D, self.__err_dBc2dT, self.__r_squared)


class RTfit():
    """This class object describes an RT fit, including all necessary parameters, attributes and methods to determine an RT fit.
    Data is read into the object as T and R arrays. Data is fitted with gaussian cdf/richards function. Fit limits for the RT fits can be defined.
    Fit can be returned as T and R arrays. Transition temperature Tc can be returned (optionally with error)"""

    def __init__(self, fit_function_type = "richards"):
        self.fit_function_type = fit_function_type
        # self.fit_function = self.richards # BUG? This line should be removed and the following 6 lines be there instead, right?
        if self.fit_function_type == "richards":
            self.fit_function = self.richards
        elif self.fit_function_type == "gauss_cdf":
            self.fit_function = self.gauss_cdf
        elif self.fit_function_type == "R_max/2":
            self.fit_function = self.linear_function
        else:
            raise ValueError('only "richards" and "gauss_cdf" as possible fitting functions')
        self.fit_function_number_of_datapoints = 1000  # amount of data points to be placed between lower and upper T values for richards/gauss_cdf fit
        self.T = 0
        self.R = 0
        self.fit_param = {'output':{}, 'start_values':{}}  # output are always the resulting fit parameters after fitting, start_values represent
                                                           # the fit parameters values used in the beginning by the fit function
        self.curve_fit_options = {}  # describes further fit options necessary for the fitting
        self.fit_covariance_matrix = {}  # matrix containing implicitly fit error
        self.__set_fit_parameters = {}  # fit parameters set by the user, default is empty

        self.T_meas_error_pc = 0.0125 # in percent, estimated interpolation error
        self.T_meas_error_std_dev = 0.02 # standard deviation of measurements at 4Kelvin
        self.R_meas_error_pc = 0.017 # relative resistance error from resistance measurement

    def read_RT_data(self, data):
        '''Input: data as dict, numpy array, tuple
        Output: sets attributes self.T and self.R
        Description: based on the type of data, the method is able to read out the data in several formats including:
        dict: {'T':x, 'R':y} ; tuple: (T,R)
        numpy.array: [[x,y], [x,y]] or [[xxx], [yyy]]'''
        try:
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
        except: raise Exception('data has not the correct format or is empty. please check passed data')

    def fit_data(self, fit_low_lim=None, fit_upp_lim=None, R_NC=None):
        '''Input: fit limits
        Output: fit parameters for one RT sweep as dictionary
        Description: reduces the array according to fit limits. Depending on fit function, sets correct fitting parameters and calculates fit with curve_fit.'''
        T, R = Tools.select_values(self.T, self.R, fit_low_lim, fit_upp_lim)
        print(self.fit_function_type)
        if T.size == 0:
            raise ValueError('chosen fit limits result in empty array. please change fit limits')
        if self.fit_function_type == "richards":
            self.__define_fitting_parameters_richards(R_NC, **self.__set_fit_parameters)
            self.fit_function = self.richards
        elif self.fit_function_type == "gauss_cdf":
            self.__define_fitting_parameters_gauss_cdf(R_NC, **self.__set_fit_parameters)
            self.fit_function = self.gauss_cdf
        elif self.fit_function_type == "R_max/2":
            # idx = self.T[bisect_left(self.R, Tools.selector(R_NC, np.max(self.R))/2)]  # index of temperature just below (or equal) where resistance reaches half its
            idx = bisect_left(self.R, Tools.selector(R_NC, np.max(self.R))/2)  # index of temperature just below (or equal) where resistance reaches half its normal conducting value
            if (idx == 0) or (idx >= np.size(T)-1):
                raise IndexError('Not enough data points for extracting R_max/2. Possibly too restricted fitting range or empty array.')
            print(idx)
            T, R = (self.T[idx-1:idx+1], self.R[idx-1:idx+1]) # reduce arrays to fit to just 2 data points for linear interpolation
            self.__define_fitting_parameters_R_max_half(**self.__set_fit_parameters)
            self.fit_function = self.linear_function
        else: raise ValueError('only "richards", "gauss_cdf" and "R_max/2" as possible fitting functions')
        #############################
        # BUG? In the next line, should it not be T and R instead of self.T and self.R because otherwise the range is not restricted to the selected one, right?
        #############################
        popt, self.fit_covariance_matrix = curve_fit(self.fit_function, self.T, self.R, list(self.fit_param['start_values'].values()), **self.curve_fit_options)
        print(popt)
        popt, self.fit_covariance_matrix = curve_fit(self.fit_function, T, R, list(self.fit_param['start_values'].values()), **self.curve_fit_options)
        print('Curve fit over selected T')
        print(popt)
        self.fit_param['output'] = {key: value for key, value in zip(self.fit_param['start_values'].keys(), popt)}
        return self.fit_param['output']

    def __define_fitting_parameters_richards(self, R_NC=None, **kwargs):
        '''Input: normal conducting resistance, optionally fitting parameters
        Output: sets starting values of fit parameters (attribute)
        Description: defines the start values if the richards function is selected. if several fits are performed, the resulting fit parameters_RTfit
        from the fit before are used as starting values'''
        R_NC = Tools.selector(R_NC, np.max(self.R))
        a,c,q = (0,1,1)
        if self.fit_param['output'] == {} and kwargs == {}:  # use calculated starting values if no fit has been already performed
            k = R_NC  # upper asymptote
            nu = 1  # affects near which asymptote maximum growth occurs (nu is always > 0)
            m = a + (k-a)/np.float_power((c+1),(1/nu))  # shift on x-axis
            t_2 = self.T[bisect_left(self.R, R_NC/2)]  # temperature where resistance reaches half its normal conducting value
            b = 1/(m-t_2) * ( np.log( np.float_power((2*(k-a)/k),nu)-c )+np.log(q) ) # growth rate
            self.fit_param['start_values'] = {'b': b, 'm': m, 'nu': nu, 'k': k}
        elif kwargs != {}:  # use parameters set by user
            self.fit_param['output'] = {}
            self.__define_fitting_parameters_richards()  # call function again to ensure missing all fitting parameters are existing
            for key, value in kwargs.items():
                if key in ['b', 'm', 'nu', 'k']:
                    self.fit_param['start_values'][key] = value  # overwrite values of keys handed over to the function in kwargs
            self.__set_fit_parameters = {}
        else:
            self.fit_param['start_values'] = self.fit_param['output']
        self.curve_fit_options = {'maxfev': 2500, 'bounds': ([-np.inf, -np.inf, -np.inf, 0.8*R_NC], [np.inf, np.inf, np.inf, 1.2*R_NC])}

    def __define_fitting_parameters_gauss_cdf(self, R_NC=None, **kwargs):
        '''Input: normal conducting resistance, optionally fitting parameters
        Output: sets starting values of fit parameters (attribute)
        Description: defines the start values if the gaussian cdf function is selected. calculates the starting values with every new self.R, self.T handed over to the class'''
        R_NC = Tools.selector(R_NC, np.max(self.R))
        if bisect_left(self.R, R_NC/2) < len(self.R):  # perform check whether the searched index for the temperature of the halfpoint value is realistic
            mean = self.T[bisect_left(self.R, R_NC/2)]  # is below the maximal length, otherwise the array is sorted the other way round and bisect left gives unrealistic indices
            sigma = self.T[bisect_left(self.R, 0.9*R_NC)]-self.T[bisect_left(self.R, 0.1*R_NC)]
        else:
            R_rev = self.R[::-1]  # turn around the array and search again for the temperature value at the halfpoint resistance, as check above was negative
            mean = self.T[bisect_left(R_rev, R_NC/2)]
            sigma = self.T[bisect_left(R_rev, 0.9*R_NC)]-self.T[bisect_left(R_rev, 0.1*R_NC)]
        if sigma < 0.01:  # ensure sigma has a good starting value
            sigma = 0.1
        self.fit_param['start_values'] = {'scaling': R_NC, 'mean': mean, 'sigma': sigma}
        if kwargs != {}:  # if fit starting values are handed over by the user, use them!
            self.__define_fitting_parameters_gauss_cdf()  # ensure there are the necessary values
            for key, value in kwargs.items():  # overwrite starting values handed over by the user
                if key in ['scaling', 'mean', 'sigma']:
                    self.fit_param['start_values'][key] = value
            self.__set_fit_parameters = {}
        self.curve_fit_options = {'maxfev': 1600, 'bounds': (-inf, inf)}

    def __define_fitting_parameters_R_max_half(self, **kwargs):
        '''Input: Optionally starting values for linear fit
        Output: sets starting values of fit parameters (attribute)
        Description: defines the start values if we want a simple linear interpolation between closest data
        points to R_max/2. Calculates the starting values with every new self.R, self.T handed over to the class'''
        self.fit_param['start_values'] = {'slope': 1.0, 'yintercept': -100}
        if kwargs != {}:  # if fit starting values are handed over by the user, use them!
            self.__define_fitting_parameters_R_max_half()  # ensure there are the necessary values
            for key, value in kwargs.items():  # overwrite starting values handed over by the user
                if key in ['slope', 'yintercept']:
                    self.fit_param['start_values'][key] = value
            self.__set_fit_parameters = {}
        self.curve_fit_options = {'maxfev': 1600, 'bounds': (-inf, inf)}

    def set_fit_parameters(self, **kwargs):  # sets the fit parameters as method for comfortness
        self.__set_fit_parameters = kwargs

    def richards(self, t,b,m,nu,k, a=0, c=1, q=1):
        return a + (k-a)/np.float_power((c+q*np.exp(-b*(t-m))),1/nu)

    def gauss_cdf(self, x, scaling, mean, sigma):
        return scaling*norm.cdf(x, mean, sigma)

    def linear_function(self, x, slope, yintercept):
        return slope*x+yintercept

    def return_RTfit(self, eval_array = None, return_eval_array = False, fit_param=None):
        '''Input: T array to be evaluated, flag if eval_array should be returned, which fit parameters to use
        Output: R array or tuple (T,R) of fitted values with the given parameters
        Description: takes or sets an evaluation array and evaluates them in the given function with the passed fitting parameters'''
        fit_param = Tools.selector(fit_param, self.fit_param['output'])
        if eval_array is None:  # set evaluation array limits as the limits of the measured self.T array
            x_low_lim = np.min(self.T)
            x_upp_lim = np.max(self.T)
            eval_array = np.linspace(x_low_lim, x_upp_lim, self.fit_function_number_of_datapoints)
        if return_eval_array:  # eval array should be returned
            return (eval_array, self.fit_function(eval_array, **fit_param))
        else:
            return self.fit_function(eval_array, **fit_param)

################ maybe insert here R_NC as parameter to give over to TC function? and then call get_TC_RMax2 with that
    def Tc(self, fit_param = None, R_NC = None):
        '''Input: fit parameters to use for determining the according Tc value of the RT-sweep
        Output: tuple (Tc, Tc_err_low, Tc_up_err): it is the transition temperature of an RT sweep
        Description: depending on the fit function, calls corresponding private method and returns Tc with the error'''
        fit_param = Tools.selector(fit_param, self.fit_param['output'])
        if self.fit_function_type == 'richards':
            Tc = self.__get_Tc_richards(fit_param)
            return (Tc, *self.__Tc_error(Tc))
        elif self.fit_function_type == 'gauss_cdf':
            Tc = self.__get_Tc_gauss_cdf(fit_param)
            return (Tc, *self.__Tc_error(Tc))
        elif self.fit_function_type == 'R_max/2':
            if R_NC == None: raise ValueError('No argument R_NC passed to RTfit.Tc although necessary for calculating Tc with R_max/2 method.')
            Tc = self.__get_Tc_R_max_half(fit_param, R_NC)
            return (Tc, *self.__Tc_error(Tc))
        else: raise ValueError('only "richards", "gauss_cdf", and "R_max/2" as possible fitting functions')

    def __get_Tc_richards(self, param):  # the halfpoint value of the richards function is analytically calculated and is then returned
        if {'b', 'm', 'nu', 'k'}.issubset(list(param.keys())):
            a,c,q = (1,1,1)
            b, m, nu, k = param.values()
            return m - 1/b*( np.log(np.float_power(2*(k-a)/k,nu)-c) + np.log(q) )  # analytical calculation of the halfpoint
        else:
            raise ValueError('no richards parameters found')

    def __get_Tc_gauss_cdf(self, param):  # the Tc of the gaussian fit is described by the expected mean mu, since this always describes the halfpoint of a gaussian cdf
        if 'mean' in param.keys():
            return param['mean']
        else: raise ValueError('no gaussian parameters found')


    def __get_Tc_R_max_half(self, param, R_NC):  # the Tc is simply given by linear interpolation
        if {'slope', 'yintercept'}.issubset(list(param.keys())):
            return (R_NC/2 - param['yintercept'])/param['slope']
        else: raise ValueError('no R_max/2 parameters found')

    def __Tc_error(self, Tc):
        '''Input: calculated Tc
        Output: Tuple (Tc_err_low, Tc_err_up)
        Description: Tc error is defined as the interval between two measured data points in which the calculated value lies, with lower and upper error respectively.
        Additionally, measurement errors are considerated'''
        T_data_from_below = self.T[bisect_left(self.T, Tc) - 1]
        T_data_from_above = self.T[bisect_left(self.T, Tc)]
        T_err_low = abs(Tc - T_data_from_below - self.T_meas_error_pc * T_data_from_below - self.T_meas_error_std_dev) # consideration of measurement errors
        T_err_up = abs(T_data_from_above + self.T_meas_error_pc * T_data_from_above + self.T_meas_error_std_dev - Tc)
        return (T_err_low, T_err_up)


class Tools():

    @staticmethod
    def selector(val, *args): # function selecting out of the passed args according to the control structure. outsourced since it was used a lot
        if val is None:       # to hand over properties, as they cannot be set as default values in the parameter definition
            return args[0]
        elif val == 'all':
            return args[1]
        else: return val

    @staticmethod
    def select_values(X, Y, fit_low_lim, fit_upp_lim):  # function reducing an x-Y relation in X dimension according to fit limits. Needed to reduce the fit area of RT sweeps and Bc2vsT relation
        XY_data = np.array([X, Y]).transpose()  # build a matrix and transpose it to have X,Y pairs
        XY_data = XY_data[XY_data[:,0].argsort()]  # sort the matrix in X dimension
        XY_data = XY_data[XY_data[:,0] >= fit_low_lim, :]  # check for lower limit
        XY_data = XY_data[XY_data[:,0] <= fit_upp_lim, :]  # check for upper limit
        T_array, Bc2_array = (XY_data[:,0], XY_data[:,1])  # create tuple of (X,Y)
        return (T_array, Bc2_array)
