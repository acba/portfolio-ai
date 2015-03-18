# -*- coding: utf-8 -*-

import numpy

try:
    import optunity
except ImportError:
    _OPTUNITY_AVAILABLE = 0
else:
    _OPTUNITY_AVAILABLE = 1

from elm.mltools import MLTools
from elm.mltools import kfold_cross_validation
from elm.mltools import time_series_cross_validation
from elm.mltools import copy_doc_of


class Mean(MLTools):
    """
        Description
    """

    regressor_name = "mean"

    default_param_w = 0

    def __init__(self, params=[]):
        super().__init__()

        if not params:
            self.param_w = self.default_param_w
        else:
            self.param_w = params[0]

    def _local_train(self, training_patterns, training_expected_targets,
                     params):

        # If params not provided, uses initialized parameters values
        if not params:
            pass
        else:
            self.param_w = params[0]

        training_time_series = numpy.concatenate([training_patterns[0][:],
                                                 training_expected_targets[0:]])

        self.mean = numpy.mean(training_time_series[-self.param_w:])

        training_predicted_targets = \
            numpy.empty(training_expected_targets.shape)

        training_predicted_targets.fill(self.mean)

        return training_predicted_targets

    def _local_test(self, testing_patterns, testing_expected_targets,
                    predicting):

            testing_predicted_targets = \
                numpy.empty(testing_expected_targets.shape)

            testing_predicted_targets.fill(self.mean)

            return testing_predicted_targets

    def search_param(self, database, dataprocess=None, path_filename=("", ""),
                     save=False, cv="ts", cv_nfolds=10, of="rmse",
                     opt_f="grid search", eval=50, print_log=True):
        """
            Search best hyperparameters for regressor based upon optunity
            algorithms.

        """

        if not _OPTUNITY_AVAILABLE:
            raise Exception("Please install 'deap' and 'optunity' library to \
                             perform search_param.")

        best_function_error = 99999.9
        best_param_w = 0

        max_range = database.shape[0]

        param_ranges = [0, max_range]

        if print_log:
            print(self.regressor_name)
            print("##### Start search #####")

        def wrapper_opt(param_w):

            param_w = int(param_w)

            if cv == "ts":
                cv_tr_error, cv_te_error = \
                    time_series_cross_validation(self, database, [param_w],
                                                 number_folds=cv_nfolds,
                                                 dataprocess=dataprocess)

            elif cv == "kfold":
                cv_tr_error, cv_te_error = \
                    kfold_cross_validation(self, database, [param_w],
                                           number_folds=cv_nfolds,
                                           dataprocess=dataprocess)

            else:
                raise Exception("Invalid type of cross-validation.")

            if of == "accuracy" or of == "hr" or of == "hr+" or of == "hr-":
                util = 1 / cv_te_error.get(of)
            else:
                util = cv_te_error.get(of)

            return util

        optimal_pars, details, _ = optunity.minimize(wrapper_opt,
                                                     solver_name=opt_f,
                                                     num_evals=eval,
                                                     param_w=param_ranges)

        if of == "accuracy" or of == "hr" or of == "hr+" or of == "hr-":
            best_function_error = 1 / details[0]
        else:
            best_function_error = details[0]
        best_param_w = optimal_pars["param_w"]

        # Mean attribute
        self.param_w = int(best_param_w)

        params = [self.param_w]

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

    def print_parameters(self):
        """
            Print parameters values.
        """

        print()
        print("Regressor Parameters")
        print()
        print("Window size: ", self.param_w)
        self.print_cv_log()


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