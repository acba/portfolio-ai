# -*- coding: utf-8 -*-

"""
    This file contains Regressor class and all developed methods.
"""

from elm import ELMRandom
from elm import ELMKernel
from .svr import SVR
from .mlp import MLP
from .mean import Mean
# from .randomwalk import RandomWalk

from elm.mltools import Error
from elm.mltools import MLTools
from elm.mltools import copy_doc_of
from elm.mltools import split_sets


class Regressor:
    """

    """

    def __init__(self, regressor_type, regressor_params=[]):

        # Regressor type
        if regressor_type is "elmrandom" or regressor_type is "r"\
                or regressor_type is "elmr":
            self.type = "elmr"
            self.regressor = ELMRandom(params=regressor_params)

        elif regressor_type is "elmkernel" or regressor_type is "k"\
                or regressor_type is "elmk":
            self.type = "elmk"
            self.regressor = ELMKernel(params=regressor_params)

        elif regressor_type is "svr" or regressor_type is "s":
            self.type = "svr"
            self.regressor = SVR(params=regressor_params)

        elif regressor_type is "mlp":
            self.type = "mlp"
            self.regressor = MLP(params=regressor_params)

        elif regressor_type is "mean":
            self.type = "mean"
            self.regressor = Mean(params=regressor_params)

        # elif regressor_type is "randomwalk":
        #     self.type = "randomwalk"
        #     self.regressor = RandomWalk(params=regressor_params)

        else:
            raise Exception("Error: Invalid type of regressor.")

        # Object responsible for scaling, splitting and transforming data
        self.dp = []

    ###########################
    #### Regressor Methods ####
    ###########################

    @copy_doc_of(MLTools._ml_train)
    def train(self, training_matrix, params=[]):
        return self.regressor.train(training_matrix, params)

    @copy_doc_of(MLTools._ml_train_it)
    def train_it(self, database_matrix, params=[], dataprocess=None,
                 sliding_window=168, k=1, search=False):
        return self.regressor.train_it(database_matrix, params, dataprocess,
                                       sliding_window, k, search)

    @copy_doc_of(MLTools._ml_predict_it)
    def predict_it(self, horizon=1, dataprocess=None):
        return self.regressor.predict_it(horizon, dataprocess)

    @copy_doc_of(MLTools._ml_test)
    def test(self, testing_matrix, predicting=False):
        return self.regressor.test(testing_matrix, predicting)

    @copy_doc_of(MLTools._ml_predict)
    def predict(self, horizon=1):
        return self.regressor.predict(horizon)

    def print_parameters(self):
        """
            Print parameters values.
        """

        self.regressor.print_parameters()

    @copy_doc_of(MLTools.save_model)
    def save_model(self, file_name):
        self.regressor.save_model(file_name)

    @copy_doc_of(MLTools.load_model)
    def load_model(self, file_name):

        self.regressor = self.regressor.load_model(file_name)
        return self

    def get_cv_flag(self):
        return self.regressor.get_cv_flag()

    def get_cv_params(self):
        return self.regressor.get_cv_params()

    def search_param(self, database, dataprocess=None, path_filename=("", ""),
                     save=False, cv="ts", cv_nfolds=10, of="rmse", kf=None,
                     f=None, opt_f="cma-es", eval=50, print_log=True):
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
                f (list of str): a list of functions to be used by the
                    search. Defaults to None, this set all available
                    functions.
                opt_f (str): Name of optunity search algorithm. Defaults to
                    "cma-es".
                eval (int): Number of steps (evaluations) to optunity algorithm.

            Each set of hyperparameters will perform a cross-validation
            method chosen by param cv.

            Available *cv* methods:
                - "ts" :func:`mltools.time_series_cross_validation()`
                    Perform a time-series cross-validation suggested by Hyndman.

                - "kfold" :func:`mltools.kfold_cross_validation()`
                    Perform a k-fold cross-validation.

            Available *of* function:
                - "accuracy", "rmse", "mape", "me".


            See Also:
                http://optunity.readthedocs.org/en/latest/user/index.html
        """

        if self.type == "elmk" or self.type == "svr":
            return self.regressor.search_param(database, dataprocess,
                                               path_filename, save, cv,
                                               cv_nfolds, of, kf, opt_f,
                                               eval, print_log)

        elif self.type == "elmr":
            return self.regressor.search_param(database, dataprocess,
                                               path_filename, save, cv,
                                               cv_nfolds, of, f, opt_f,
                                               eval, print_log)

        elif self.type == "mlp":
            return self.regressor.search_param(database, dataprocess,
                                               path_filename, save, cv,
                                               cv_nfolds, of, kf, opt_f,
                                               eval, print_log)

        elif self.type == "mean":
            opt_f = "grid search"
            return self.regressor.search_param(database, dataprocess,
                                               path_filename, save, cv,
                                               cv_nfolds, of, opt_f, eval,
                                               print_log)

        else:
            raise Exception("Unavailable search_param method.")

    def auto_ts(self, data, dataprocess=None, predict_horizon=5):
        """
            Auto
        """

        ########################
        #### Pre Processing ####
        ########################

        # 1 - Split data matrix in training and testing sets

        self.dp = dataprocess

        train_set, test_set = split_sets(data, n_test_samples=predict_horizon)

        # 2 - Scale and Apply a transformation to the data
        train_set, test_set = self.dp.auto(train_set, test_set)

        # 3 - Starts a brute force search through all parameters range space
        self.search_param(database=data, dataprocess=self.dp)

        # 4 - Use the best parameter found and train, test and predict the
        # desired data
        tr_result = self.train(training_matrix=train_set)
        te_result = self.test(testing_matrix=test_set)
        predicted_targets = self.predict(horizon=predict_horizon)

        pr_result = Error(expected=te_result.expected_targets,
                          predicted=predicted_targets)

        ########################
        #### Pos Processing ####
        ########################

        # print("Training Errors:")
        # tr_result.print_errors()
        #
        print("Testing Errors:")
        te_result.print_errors()

        print("Prediction Errors:")
        pr_result.print_errors()
        # pr_result.print_values()

        return tr_result, te_result, pr_result
