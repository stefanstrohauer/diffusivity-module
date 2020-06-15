from import_lib import *

class DiffusivityMeasurement:
    def __init__(self, filename): # to see why it is cleaner to create new object
                                  # https://stackoverflow.com/questions/29947810/reusing-instances-of-objects-vs-creating-new-ones-with-every-update
        self.filename = filename
        self.key_w_data = 'data'
        self._raw_data = self.__import_file_mat()
        self.index_t = 8
        self.index_B = 9
        self.index_R = 10
        self.time_sweeps = [i[self.index_t] for i in self._raw_data]
        self.T_sweeps = self.__time_temperature_mapping()
        self.B_sweeps = [i[self.index_B] for i in self._raw_data]
        self.R_sweeps = [i[self.index_R] for i in self._raw_data]
        self.__RT_sweeps_per_B = {}

        self.diffusivity = 0
        self.diffusivity_err = 0
        self.default_B_array = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.B_bin_size = 0.01
        self.sheet_resistance_geom_factor = 4.53
        self.sheet_resistance = 0

        self.__rearrange_B_sweeps()

    def __import_file_mat(self):
        """reads string with filename and returns .mat data file as list of lists"""
        return io.loadmat(self.filename, squeeze_me=True)[self.key_w_data]

    def __rearrange_B_sweeps(self):
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

    def R_vs_T(self, B_values=None, err=False):
        if B_values is None:
            B_values = self.default_B_array
        B_values = self.__round_to_base(B_values, base=self.B_bin_size)
        if isinstance(B_values, int, float):
            return (self.__RT_sweeps_per_B[B_values]['T'], self.__RT_sweeps_per_B[B_values]['R'])
        else:
            print('hallo')
            T_B_dict = {key:value['T'] for key, value in self.__RT_sweeps_per_B.items()}
            R_B_dict = {key:value['R'] for key, value in self.__RT_sweeps_per_B.items()}
            return (T_B_dict, R_B_dict)

    def R_vs_B(self, B_sweep_index, err=False):
        # TODO: implement when err=True
        return (self.B_sweeps[B_sweep_index], self.R_sweeps[B_sweep_index])

    def Bc2_vs_T(arg):
        pass

    def calc_diffusivity(arg):
        pass


    class Bc2vsTfit(object):
        """docstring for Bc2vsTfit."""

        def __init__(self, arg):
            super(Bc2vsTfit, self).__init__()
            self.arg = arg

        def linear_fit(arg):
            pass


class RTfit(object):
    """docstring for RTfit."""

    def __init__(self, arg):
        super(RTfit, self).__init__()
        self.arg = arg

    def fit_function_parameters(arg):
        pass

    def fit_function(arg):
        pass

    def Tc(arg):
        pass


T=DiffusivityMeasurement('./testing_meas/200212_200109A_diffusweep.mat')
T.R_vs_T()

print(b)
