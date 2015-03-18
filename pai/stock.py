# -*- coding: utf-8 -*-

"""
    This file contains Stock class and all developed methods
"""


import numpy
from .data import Data

try:
    from matplotlib import pyplot as plt
except ImportError:
    _MATPLOTLIB_AVAILABLE = 0
else:
    _MATPLOTLIB_AVAILABLE = 1


class Stock:
    """
        Classe que contem os atributos de cada ação
    """

    def __init__(self, name, time_series_data=None, file_name=None,
                 gen_param=None):
        self.name = name
        self.has_stock_values = False

        self.time_series_data = time_series_data

        # Declared variables

        self.expected_return = []
        self.returns = []
        self.data_matrix = []

        self.risk = 0

        if self.time_series_data is None:

            # If a file containing data is provided
            if file_name is not None:
                self.get_data(file_name)
            else:
                self.get_data(gen_param=gen_param)
        else:
            self.has_stock_values = True

    def plot_stock(self, show=True, save=False, path_filename=("", "")):
        """ Plot stock values.

            Args:
                show -
                save -

            Returns:
                None

        """

        if not _MATPLOTLIB_AVAILABLE:
            raise Exception("Please install 'matplotlib' package to plot data.")

        if self.has_stock_values:
            plt.figure(figsize=(24.0, 12.0))
            plt.plot(self.time_series_data)

            plt.xlabel('Time')
            plt.ylabel('stock values')
            plt.title(self.name)

            if path_filename[0] is "":
                if path_filename[1] is "":
                    filename = self.name + "_stock_values.png"
                else:
                    filename = path_filename[1] + "_" + self.name + \
                               "_stock_values.png"
            else:
                if path_filename[1] is "":
                    filename = path_filename[0] + "/" + self.name + \
                           "_stock_values.png"
                else:
                    filename = path_filename[0] + "/" + path_filename[1] + \
                               "_" + self.name + "_stock_values.png"
            if save:
                plt.savefig(filename, dpi=100)
            if show:
                plt.show()

            plt.close()

        else:
            print("Error: Must set stock values before trying to plot.")
            return

    def plot_return(self, show=True, save=False, path_filename=("", "")):
        """ Plot returns from stock.

            Args:
                show -
                save -

            Returns:
                None

        """

        if not _MATPLOTLIB_AVAILABLE:
            raise Exception("Please install 'matplotlib' package to plot data.")

        if self.has_stock_values:
            plt.figure(figsize=(24.0, 12.0))
            plt.plot(self.returns, label="Return Values")
            plt.axhline(self.expected_return, color='r',
                        label="Expected Return")

            msg = self.name + "\n$\mu = $" + str(numpy.around(
                self.expected_return, decimals=6)) + "\n$\sigma = $" + \
                str(numpy.around(self.risk, decimals=6))

            plt.xlabel('Time')
            plt.ylabel('return values')
            plt.title(msg)
            plt.legend(loc='upper right', shadow=True)

            if path_filename[0] is "":
                if path_filename[1] is "":
                    filename = self.name + "_return_values.png"
                else:
                    filename = path_filename[1] + "_" + self.name + \
                               "_return_values.png"
            else:
                if path_filename[1] is "":
                    filename = path_filename[0] + "/" + self.name + \
                               "_return_values.png"
                else:
                    filename = path_filename[0] + "/" + path_filename[1] + \
                               "_" + self.name + "_return_values.png"

            if save:
                plt.savefig(filename, dpi=100)
            if show:
                plt.show()

            plt.close()

        else:
            print("Error: Must set stock values before trying to plot.")
            return

    def expected_return_mean(self, samples=0, show=False):
        """ Calculates expected return as a mean from last "samples" return
            values.

            If samples argument is not provided, default value is 0, so it will
            perform mean calculation from all return values.

            Args:
                samples - number of last return values to calculate mean
                expected return (Default: 0)
                show - boolean that shows calculated expected return (
                Default: False)

            Returns:
                expected_return - mean from last "samples" returns.

        """

        # Mean from last "samples" return values
        expected_return = numpy.mean(self.returns[-samples:])

        if show:
            print("Calculated mean expected return from ", self.name, " is ",
                  expected_return)

        return expected_return

    def risk_std(self, samples=0, show=False):
        """ Calculates risk as a standard deviation from last "samples" return
            values.

            Args:
                samples - number of last return values to calculate risk as
                standard deviation (Default: 0)
                show - boolean that shows calculated risk (Default: False)

            Returns:
                risk - standard deviation from last "samples" returns.

        """

        # Standard deviation from last "samples" return values
        risk = numpy.std(self.returns[-samples:])

        if show:
            print("Calculated std risk from ", self.name, " is ", risk)

        return risk

    def calculate_return(self):
        """ Calculate return from stock values.

            Args:

            Returns:

        """

        return_length = len(self.time_series_data) - 1
        self.returns = numpy.empty(return_length)

        for i in range(return_length):
            self.returns[i] = \
                (self.time_series_data[i + 1] - self.time_series_data[i]) / \
                self.time_series_data[i]

        self.expected_return = self.expected_return_mean()
        self.risk = self.risk_std()

    def get_data(self, file_name=None, reverse=False, gen_param=None):
        """ Set stock values provided by 'file_name' data.

            Args:
                file_name - File name of data.
                reverse - If oldest stock values are provided at the end of
                    file.

            Returns:

        """

        if file_name is not None:
            if reverse is True:
                self.time_series_data = Data().get_data(file_name)[::-1]
            else:
                self.time_series_data = Data().get_data(file_name)[::]
        else:
            # Default artificial data
            if gen_param is None:
                self.time_series_data = \
                    Data().generate_data(size=1000, typo="r")
            else:
                if 'size' not in gen_param or 'type' not in gen_param or \
                        'params' not in gen_param:

                    print("Error: gen_param must be a dictionary type "
                          "containing 'size', 'type' and 'params' keys.")
                    return
                else:
                    self.time_series_data = \
                        Data().generate_data(size=gen_param['size'],
                                             typo=gen_param['type'],
                                             args=gen_param['params'])

        self.has_stock_values = True
        self.calculate_return()

    def create_database(self, window_size=30, k=1, series_type="return"):
        """

            Args:
                function_type -

            Returns:
                best_param

        """

        if window_size < 1:
            window_size = 1

        if series_type is "return":
            time_series = self.returns
        elif series_type is "stock":
            time_series = self.time_series_data
        else:
            print("Error: Invalid type of time series. Choose either "
                  "'return' or 'stock.")
            return

        time_series_size = time_series.shape[0]

        if window_size + k > time_series_size:
            print("Error : parameters accessed invalid index. 'window_size' "
                  "or 'k' too high.")
            return

        number_of_patterns = time_series_size - window_size - k + 1

        self.data_matrix = numpy.empty((number_of_patterns, window_size + 1))

        for i in range(number_of_patterns):
            self.data_matrix[i, :] = numpy.concatenate(
                [time_series[i + window_size + k - 1].reshape(1, -1),
                 time_series[i:i + window_size].reshape(1, -1)], axis=1)

    def get_database(self):

        return self.data_matrix

    def get_stock_name(self):
        return self.name

    def get_return_series(self):
        return self.returns

    def get_time_series(self):
        return self.time_series_data


def ts_to_matrix(ts, ws, ta=1, k=1):

    # print()
    # print("ts_to_matrix")
    n_lines = len(ts) - ws - ta + 1

    # print("n_lines ", n_lines)

    datamatrix = numpy.empty((n_lines, ws + ta))
    # print("datamatrix ", datamatrix.shape)

    for i in range(n_lines):
        datamatrix[i, :] = numpy.concatenate([ts[i + ws + k - 1].reshape(1, -1),
                                              ts[i:i + ws].reshape(1, -1)],
                                             axis=1)
    # print("datamatrix ", datamatrix.shape)
    # print()
    return datamatrix