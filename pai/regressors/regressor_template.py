# -*- coding: utf-8 -*-

from regressors.regressortools import RegressorTools, BestParam

class TemplateRegressor(RegressorTools):
    """
        Description
    """

    regressor_name = "name"
    default_params = []

    def __init__(self, params=[]):
        super().__init__()

        # Initialized parameters values
        if not params:
            self.params = self.default_params
        else:
            self.params = params

    def _local_train(self, training_patterns, training_expected_targets,
                  params):

        # If params not provided, uses initialized parameters values
        if not params:
            pass
        else:
            self.params = params


        training_predicted_targets = []

        return training_predicted_targets

    def _local_test(self, testing_patterns, testing_expected_targets):

            testing_predicted_targets = []

            return testing_predicted_targets

    def search_best_param(self, database, scale=None, path_filename=("", ""),
                          ranges=None, save=False):

        best_param = []

        return best_param

