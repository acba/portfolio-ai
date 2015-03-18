# -*- coding: utf-8 -*-

"""
    This file contains Data class and all developed methods
"""

# Python2 support
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy


class Data:
    """ Classe responsável por repassar para a camada superior os dados pedidos
    """

    def __init__(self):
        pass

    def generate_data(self, size=1000, typo="r", args=None):
        """ Generate several types of data, constant function, sine function
            and random walk .

            Args:
                size -
                typo -
                args -

            Returns:
                None

        """

        # Random Walk
        if typo == "r" or typo == "random" or typo == 0 or typo == "randomwalk":
            if args is None:
                mean = 0
                std = 1

            else:
                mean = args[0]
                std = args[1]

            return numpy.cumsum(numpy.random.normal(mean, std, size))

        # Constant function
        elif typo == "c" or typo == 1:
            if args is None:
                value = 1

            else:
                value = args

            return value * numpy.ones(size)

        # Sine function
        elif typo == "s" or typo == "sin" or typo == 2 or typo is "sine":
            if len(args) == 1:
                return numpy.sin(numpy.linspace(0, 2 * numpy.pi, size)) + \
                       args[0]

            elif len(args) == 2:
                return numpy.sin(
                    numpy.linspace(0, args[1] * 2 * numpy.pi, size)) + args[0]

            elif len(args) == 3:
                return args[2] * numpy.sin(
                    numpy.linspace(0, args[1] * 2 * numpy.pi, size)) + args[0]

            else:
                print("Data.py - generateData")
                return
                # raise error

    def get_data(self, file_name):
        """ Get data from text file.

            Args:
                file_name - file name.

            Returns:
                An numpy.ndarray containing the read data.

        """

        return numpy.loadtxt(file_name, dtype=float)
