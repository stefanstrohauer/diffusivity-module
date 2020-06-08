from import_lib import *

class DiffusivityMeasurement:
    def __init__(self, filename): # to see why it is cleaner to create new object
                                  # https://stackoverflow.com/questions/29947810/reusing-instances-of-objects-vs-creating-new-ones-with-every-update
        self.filename = filename
        self.key_w_data = 'data'
        self._raw_data = self.__import_file_mat()

        self.diffusivity = 0
        self.diffusivity_err = 0
        self.default_B_array = []
        self.B_bin_size = 0.01
        self.sheet_resistance_geom_factor = 4.53
        self.sheet_resistance = 0


    def __import_file_mat(self):
        """reads string with filename and returns .mat data file as list of lists"""
        return io.loadmat(self.filename, squeeze_me=True)[self.key_w_data]

    def time_temperature_mapping():


    def R_vs_B(B_sweep_index, err=False):
        # TODO: implement when err=True
        B_array_position = 9
        R_array_position = 10
        sweep = self._raw_data[B_sweep_index]
        return (sweep[B_array_position], sweep[R_array_position])

    def R_vs_T(arg):
        pass

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
