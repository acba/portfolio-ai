# -*- coding: utf-8 -*-

# Python2 support
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import configparser
import ast
import sys

from elm.mltools import MLTools
from elm.mltools import kfold_cross_validation
from elm.mltools import time_series_cross_validation
from elm.mltools import copy_doc_of

try:
    from sklearn.svm import SVR as SKSVR
except ImportError:
    _SKLEARN_AVAILABLE = 0
else:
    _SKLEARN_AVAILABLE = 1

try:
    import optunity
except ImportError:
    _OPTUNITY_AVAILABLE = 0
else:
    _OPTUNITY_AVAILABLE = 1


# Find configuration file
from pkg_resources import Requirement, resource_filename
_SVR_CONFIG = resource_filename(Requirement.parse("portfolio-ai"),
                                "pai/regressors/svr.cfg")


class SVR(MLTools):
    """
        SVR
    """

    def __init__(self, params=[]):
        super(self.__class__, self).__init__()

        if not _SKLEARN_AVAILABLE:
            raise Exception("Please install 'sklearn' package to use SVR.")

        self.regressor_name = "svr"

        self.available_kernel_functions = ["rbf", "linear", "poly"]

        # Default parameters
        self.default_param_c = 9
        self.default_param_kernel_function = "rbf"
        self.default_param_kernel_params = [-10]
        self.default_param_e = 0.1

        # Initialized parameters values
        if not params:
            self.param_c = self.default_param_c
            self.param_kernel_function = self.default_param_kernel_function
            self.param_kernel_params = self.default_param_kernel_params
            self.param_e = self.default_param_e
        else:
            self.param_c = params[0]
            self.param_kernel_function = params[1]
            self.param_kernel_params = params[2]
            self.param_e = params[3]

        self.svr = []

    # ########################
    # Private Methods
    # ########################

    def _local_train(self, training_patterns, training_expected_targets,
                     params):

        # If params not provided, uses initialized parameters values
        if not params:
            pass
        else:
            self.param_c = params[0]
            self.param_kernel_function = params[1]
            self.param_kernel_params = params[2]
            self.param_e = params[3]

        if self.param_kernel_function == "rbf":
            self.svr = SKSVR(kernel='rbf', C=2**self.param_c,
                             gamma=2**self.param_kernel_params[0],
                             epsilon=self.param_e, cache_size=2000)

        elif self.param_kernel_function == "linear":
            self.svr = SKSVR(kernel='linear', C=2**self.param_c,
                             epsilon=self.param_e, cache_size=2000)

        elif self.param_kernel_function == "poly":

            degree = round(self.param_kernel_params[1])

            self.svr = SKSVR(kernel='poly', C=2**self.param_c,
                             coef0=self.param_kernel_params[0],
                             degree=degree, epsilon=self.param_e,
                             cache_size=2000)

        else:
            print("Error: Invalid kernel function.")

        self.svr.fit(training_patterns, training_expected_targets)

        training_predicted_targets = self.svr.predict(training_patterns)

        return training_predicted_targets

    def _local_test(self, testing_patterns, testing_expected_targets,
                    predicting):

        testing_predicted_targets = self.svr.predict(testing_patterns)

        return testing_predicted_targets

    # ########################
    # Public Methods
    # ########################

    def get_available_kernel_functions(self):
        """
            Return available kernel functions.
        """

        return self.available_kernel_functions

    def print_parameters(self):
        """
            Print parameters values.
        """

        print()
        print("Regressor Parameters")
        print()
        print("Regularization coefficient: ", self.param_c)
        print("Kernel Function: ", self.param_kernel_function)
        print("Kernel parameters: ", self.param_kernel_params)
        print("Epsilon: ", self.param_e)
        self.print_cv_log()

    def search_param(self, database, dataprocess=None, path_filename=("", ""),
                     save=False, cv="ts", cv_nfolds=10, of="rmse", kf=None,
                     opt_f="cma-es", eval=50, print_log=True):
        """
            Search best hyperparameters for classifier/regressor based on
            optunity algorithms.

            Arguments:
                database (numpy.ndarray): a matrix containing all patterns
                    that will be used for training/testing at some
                    cross-validation method.
                dataprocess (DataProcess): an object that will pre-process
                    database before training. Defaults to None.
                path_filename (tuple): *TODO*.
                save (bool): *TODO*.
                cv (str): Cross-validation method. Defaults to "ts".
                cv_nfolds (int): Number of Cross-validation folds. Defaults
                    to 10.
                of (str): Objective function to be minimized at
                    optunity.minimize. Defaults to "rmse".
                kf (list of str): a list of kernel functions to be used by
                    the search. Defaults to None, this set all available
                    functions.
                opt_f (str): Name of optunity search algorithm. Defaults to
                    "cma-es".
                eval (int): Number of steps (evaluations) to optunity algorithm.


            Each set of hyperparameters will perform a cross-validation
            method chosen by param cv.


            Available *cv* methods:
                - "ts" :func:`mltools.time_series_cross_validation()`
                    Perform a time-series cross-validation suggested by Hydman.

                - "kfold" :func:`mltools.kfold_cross_validation()`
                    Perform a k-fold cross-validation.

            Available *of* function:
                - "accuracy", "rmse", "mape", "me".


            See Also:
                http://optunity.readthedocs.org/en/latest/user/index.html
        """

        if not _OPTUNITY_AVAILABLE:
            raise Exception("Please install 'deap' and 'optunity' library to \
                             perform search_param.")

        if kf is None:
            search_kernel_functions = self.available_kernel_functions
        elif type(kf) is list:
            search_kernel_functions = kf
        else:
            raise Exception("Invalid format for argument 'kf'.")

        if print_log:
            print(self.regressor_name)
            print("##### Start search #####")

        config = configparser.ConfigParser()

        if sys.version_info < (3, 0):
            config.readfp(open(_SVR_CONFIG))
        else:
            config.read_file(open(_SVR_CONFIG))

        best_function_error = 99999.9
        temp_error = best_function_error

        best_param_c = 0
        best_param_kernel_function = ""
        best_param_kernel_param = []
        best_param_e = 0

        for kernel_function in search_kernel_functions:

            if sys.version_info < (3, 0):
                svr_c_range = ast.literal_eval(config.get("DEFAULT",
                                                          "svr_c_range"))

                n_parameters = config.getint(kernel_function, "kernel_n_param")
                kernel_p_range = \
                    ast.literal_eval(config.get(kernel_function,
                                                "kernel_params_range"))

            else:
                kernel_config = config[kernel_function]

                svr_c_range = ast.literal_eval(kernel_config["svr_c_range"])

                n_parameters = int(kernel_config["kernel_n_param"])
                kernel_p_range = \
                    ast.literal_eval(kernel_config["kernel_params_range"])

            param_ranges = [[svr_c_range[0][0], svr_c_range[0][1]]]
            for param in range(n_parameters):
                    param_ranges.append([kernel_p_range[param][0],
                                         kernel_p_range[param][1]])

            def wrapper_0param(param_c, param_e):
                """
                    Wrapper for objective function.
                """

                if cv == "ts":
                    cv_tr_error, cv_te_error = \
                        time_series_cross_validation(self, database,
                                                     [param_c,
                                                      kernel_function,
                                                      list([]),
                                                      param_e],
                                                     number_folds=cv_nfolds,
                                                     dataprocess=dataprocess)

                elif cv == "kfold":
                    cv_tr_error, cv_te_error = \
                        kfold_cross_validation(self, database,
                                               [param_c,
                                                kernel_function,
                                                list([]),
                                                param_e],
                                               number_folds=cv_nfolds,
                                               dataprocess=dataprocess)

                else:
                    raise Exception("Invalid type of cross-validation.")

                if of == "accuracy" or of == "hr" or of == "hr+" or of == "hr-":
                    util = 1 / cv_te_error.get(of)
                else:
                    util = cv_te_error.get(of)

                # print("c:", param_c, "util: ", util)
                return util

            def wrapper_1param(param_c, param_kernel, param_e):
                """
                    Wrapper for optunity.
                """

                if cv == "ts":
                    cv_tr_error, cv_te_error = \
                        time_series_cross_validation(self, database,
                                                     [param_c,
                                                      kernel_function,
                                                      list([param_kernel]),
                                                      param_e],
                                                     number_folds=cv_nfolds,
                                                     dataprocess=dataprocess)

                elif cv == "kfold":
                    cv_tr_error, cv_te_error = \
                        kfold_cross_validation(self, database,
                                               [param_c,
                                                kernel_function,
                                                list([param_kernel]),
                                                param_e],
                                               number_folds=cv_nfolds,
                                               dataprocess=dataprocess)

                else:
                    raise Exception("Invalid type of cross-validation.")

                if of == "accuracy" or of == "hr" or of == "hr+" or of == "hr-":
                    util = 1 / cv_te_error.get(of)
                else:
                    util = cv_te_error.get(of)

                # print("c:", param_c, " gamma:", param_kernel, "util: ", util)
                return util

            def wrapper_2param(param_c, param_kernel1, param_kernel2, param_e):
                """
                    Wrapper for optunity.
                """

                if cv == "ts":
                    cv_tr_error, cv_te_error = \
                        time_series_cross_validation(self, database,
                                                     [param_c,
                                                      kernel_function,
                                                      list([param_kernel1,
                                                            param_kernel2]),
                                                      param_e],
                                                     number_folds=cv_nfolds,
                                                     dataprocess=dataprocess)

                elif cv == "kfold":
                    cv_tr_error, cv_te_error = \
                        kfold_cross_validation(self, database,
                                               [param_c,
                                                kernel_function,
                                                list([param_kernel1,
                                                      param_kernel2]),
                                                param_e],
                                               number_folds=cv_nfolds,
                                               dataprocess=dataprocess)

                else:
                    raise Exception("Invalid type of cross-validation.")

                if of == "accuracy" or of == "hr" or of == "hr+" or of == "hr-":
                    util = 1 / cv_te_error.get(of)
                else:
                    util = cv_te_error.get(of)

                # print("c:", param_c, " param1:", param_kernel1,
                #       " param2:", param_kernel2, "util: ", util)
                return util

            if kernel_function == "linear":
                optimal_parameters, details, _ = \
                    optunity.minimize(wrapper_0param,
                                      solver_name=opt_f,
                                      num_evals=eval,
                                      param_c=param_ranges[0],
                                      param_e=[0, 1])

            elif kernel_function == "rbf":
                optimal_parameters, details, _ = \
                    optunity.minimize(wrapper_1param,
                                      solver_name=opt_f,
                                      num_evals=eval,
                                      param_c=param_ranges[0],
                                      param_kernel=param_ranges[1],
                                      param_e=[0, 1])

            elif kernel_function == "poly":
                optimal_parameters, details, _ = \
                    optunity.minimize(wrapper_2param,
                                      solver_name=opt_f,
                                      num_evals=eval,
                                      param_c=param_ranges[0],
                                      param_kernel1=param_ranges[1],
                                      param_kernel2=param_ranges[2],
                                      param_e=[0, 1])
            else:
                raise Exception("Invalid kernel function.")

            # Save best kernel result
            if details[0] < temp_error:
                temp_error = details[0]

                if of == "accuracy" or of == "hr" or of == "hr+" or of == "hr-":
                    best_function_error = 1 / temp_error
                else:
                    best_function_error = temp_error

                best_param_kernel_function = kernel_function
                best_param_c = optimal_parameters["param_c"]
                best_param_e = optimal_parameters["param_e"]

                if best_param_kernel_function == "linear":
                    best_param_kernel_param = []
                elif best_param_kernel_function == "rbf":
                    best_param_kernel_param = [optimal_parameters["param_kernel"]]
                elif best_param_kernel_function == "poly":
                    best_param_kernel_param = \
                        [optimal_parameters["param_kernel1"],
                         round(optimal_parameters["param_kernel2"])]
                else:
                    raise Exception("Invalid kernel function.")

                # print("best: ", best_param_kernel_function,
                #       best_function_error, best_param_c, best_param_kernel_param)

            if print_log:
                if of == "accuracy" or of == "hr" or of == "hr+" or of == "hr-":
                    print("Kernel function: ", kernel_function,
                          " best cv value: ", 1/details[0])
                else:
                    print("Kernel function: ", kernel_function,
                          " best cv value: ", details[0])

        # SVR attribute
        self.param_c = best_param_c
        self.param_kernel_function = best_param_kernel_function
        self.param_kernel_params = best_param_kernel_param
        self.param_e = best_param_e

        params = [self.param_c,
                  self.param_kernel_function,
                  self.param_kernel_params,
                  self.param_e]

        # MLTools Attribute
        self.has_cv = True
        self.cv_name = cv
        self.cv_error_name = of
        self.cv_best_error = best_function_error
        self.cv_best_params = params

        if print_log:
            print("##### Search complete #####")
            self.print_parameters()

        return params

    @copy_doc_of(MLTools._ml_train)
    def train(self, training_matrix, params=[]):
        return self._ml_train(training_matrix, params)

    @copy_doc_of(MLTools._ml_test)
    def test(self, testing_matrix, predicting=False):
        return self._ml_test(testing_matrix, predicting)

    @copy_doc_of(MLTools._ml_predict)
    def predict(self, horizon=1):
        return self._ml_predict(horizon)

    @copy_doc_of(MLTools._ml_train_it)
    def train_it(self, database_matrix, params=[], dataprocess=None,
                 sliding_window=168, k=1, search=False):
        return self._ml_train_it(database_matrix, params, dataprocess,
                                 sliding_window, k, search)

    @copy_doc_of(MLTools._ml_predict_it)
    def predict_it(self, horizon=1, dataprocess=None):
        return self._ml_predict_it(horizon, dataprocess)
