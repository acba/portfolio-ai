# -*- coding: utf-8 -*-

import numpy

from elm.mltools import MLTools


class RandomWalk(MLTools):
    """
        Description
    """

    regressor_name = "randomwalk"
    number_of_parameters = 2

    default_param_mean = 0
    default_param_std = 1

    def __init__(self, params=[]):
        super().__init__()

        if not params:
            self.param_mean = self.default_param_mean
            self.param_std = self.default_param_std
        else:
            self.param_mean = params[0]
            self.param_std = params[1]

    def _local_train(self, training_patterns, training_expected_targets,
                  params):

        # If params not provided, uses initialized parameters values
        if not params:
            pass
            # self.param_std = numpy.std(training_expected_targets)
        else:
            self.param_mean = params[0]
            self.param_std = params[1]

        first_target = training_expected_targets[0]
        random_walk = numpy.random.normal(self.param_mean,
                                          self.param_std,
                                          training_expected_targets.size)

        training_predicted_targets = first_target + numpy.cumsum(random_walk)

        # Save last target to be the start of the random walk
        self.last_training_target = training_expected_targets[-1]
        self.predict_base = training_expected_targets[-1]

        return training_predicted_targets

    def _local_test(self, testing_patterns, testing_expected_targets,
                    predicting):

            random_walk = numpy.random.normal(self.param_mean,
                                              self.param_std,
                                              testing_expected_targets.size)

            if predicting:
                x_t_plus_1 = self.predict_base + + numpy.cumsum(random_walk)
                testing_predicted_targets = x_t_plus_1
                self.predict_base = x_t_plus_1
            else:
                testing_predicted_targets = \
                    self.last_training_target + numpy.cumsum(random_walk)

            return testing_predicted_targets

    def search_best_param(self, database, scale=None, path_filename=("", ""),
                          ranges=None, save=False):

        best_param = []

        return best_param

