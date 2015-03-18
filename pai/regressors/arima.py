# -*- coding: utf-8 -*-

import numpy
import sys
from regressors.regressortools import RegressorTools, Error, BestParam

class ARIMA(RegressorTools):
    """ ARIMA

    """

    default_param_kernel_function = "rbf"
    default_param_c = 2 ** 9
    default_param_g = 2 ** -10
    default_param_e = 0.1

    def __init__(self, params=[]):
        super().__init__()

        self.svr = []

        # Initialized parameters values
        if not params:
            self.param_kernel_function = self.default_param_kernel_function
            self.param_c = self.default_param_c
            self.param_g = self.default_param_g
            self.param_e = self.default_param_e
        else:
            self.param_kernel_function = params[0]
            self.param_c = params[1]
            self.param_g = params[2]
            self.param_e = params[3]

    # ########################
    # Private Methods
    # ########################

    # ########################
    # Public Methods
    # ########################

    def train(self, training_matrix, params=[]):
        """ Calculate output_weight values needed to predict data.

            If training_matrix is provided, this method will use it to
            perform  output_weight calculation. Else, it will use the
            default value provided at 'set_database' method.

            If params is provided, this method will use it to perform
            output_weight calculation. Else, it will use the default value
            provided at object initialization.

            If number of number of patterns for training were much bigger
            than the dimension problem 'optimized' can and should be True.

            Args:
                training_matrix -
                params -

            Returns:
                Dictionary with expected and predicted values.

        """

        training_patterns = training_matrix[:, 1:]
        training_expected_targets = training_matrix[:, 0]

        # If params not provided, uses initialized parameters values
        if not params:
            pass
            # param_kernel_function = self.param_kernel_function
            # param_c = self.param_c
            # param_g = self.param_g
            # param_e = self.param_e
        else:
            self.param_kernel_function = params[0]
            self.param_c = params[1]
            self.param_g = params[2]
            self.param_e = params[3]

        if self.param_kernel_function is "rbf":
            self.svr = SKSVR(kernel='rbf', C=self.param_c, gamma=self.param_g,
                             epsilon=self.param_e, cache_size=2000)
        else:
            print("Error: Invalid kernel function.")

        self.svr.fit(training_patterns, training_expected_targets)

        training_predicted_targets = self.svr.predict(training_patterns)

        # Go back to original data scale and calculates the errors measures
        if self.has_scaled:
            training_predicted_targets = \
                self.rescale(training_predicted_targets)
            training_expected_targets = self.rescale(training_expected_targets)

        training_errors = Error(training_expected_targets,
                                training_predicted_targets,
                                regressor_name="svr")

        # Save last pattern for posterior predictions
        self.last_training_pattern = training_matrix[-1, :]
        self.has_trained = True

        return training_errors

    def test(self, testing_matrix, predicting=False):
        """ Calculate testing predicted values based upon previous training.

            If params is provided, this method will use it to perform the
            prediction. Else, it will use the parameters values provided at
            object initialization.

            Args:
                testing_matrix -
                params -

            Returns:
                Dictionary with expected and predicted values.

        """

        testing_patterns = testing_matrix[:, 1:]
        testing_expected_targets = testing_matrix[:, 0].reshape(-1, 1)

        # If params not provided, uses initialized parameters values
        # if not params:
        #     param_kernel_function = self.param_kernel_function
        #     param_c = self.param_c
        #     param_g = self.param_g
        #     param_e = self.param_e
        # else:
        #     print("Warning: These parameters won't be used, it will use the "
        #           "training session parameters.")
        #     param_kernel_function = params[0]
        #     param_c = params[1]
        #     param_g = params[2]
        #     param_e = params[3]

        testing_predicted_targets = self.svr.predict(testing_patterns)

        # Go back to original data scale and calculates the errors measures
        if self.has_scaled and not predicting:
            testing_predicted_targets = self.rescale(testing_predicted_targets)
            testing_expected_targets = self.rescale(testing_expected_targets)

        testing_errors = Error(testing_expected_targets,
                               testing_predicted_targets,
                               regressor_name="svr")

        return testing_errors

    def predict(self, horizon=30):
        """ Predict next targets based on previous training data.

            Args:
                horizon -
                params -

            Returns:
                best_param

        """

        if not self.has_trained:
            print("Error: Train before predict.")
            return

        # If params not provided, uses initialized parameters values
        # if not params:
        #     param_kernel_function = self.param_kernel_function
        #     param_c = self.param_c
        #     param_g = self.param_g
        #     param_e = self.param_e
        # else:
        #     print("Warning: These parameters won't be used, it will use the "
        #           "training session parameters.")
        #     param_kernel_function = params[0]
        #     param_c = params[1]
        #     param_g = params[2]
        #     param_e = params[3]

        # Create first new pattern
        new_pattern = numpy.hstack(
            [self.last_training_pattern[2:], self.last_training_pattern[0]])

        # Create a fake target
        new_pattern = numpy.insert(new_pattern, 0, 1).reshape(1, -1)

        predicted_targets = numpy.zeros((horizon, 1))

        for t_counter in range(horizon):
            te_errors = \
                self.test(new_pattern, predicting=True)

            predicted_value = te_errors.predicted_targets
            predicted_targets[t_counter] = predicted_value

            # Create a new pattern including the actual predicted value
            new_pattern = numpy.hstack(
                [new_pattern[0, 2:], numpy.squeeze(predicted_value)])

            # Create a fake target
            new_pattern = numpy.insert(new_pattern, 0, 1).reshape(1, -1)

        if self.has_scaled:
            predicted_targets = \
                self.rescale(predicted_targets)

        return predicted_targets

    def search_best_param(self, database, scale=None, path_filename=("", ""),
                          ranges=None, save=False):
        """ Brute force search for parameters with minimum RMSE.

            Args:
                function_type -

            Returns:
                best_param

        """

        # if param_function is "default":
        param_kernel_function = "rbf"
        # else:
        #     print("Error: Invalid or unavailable kernel function.")
        #     return

        if ranges is None:
            c_range = numpy.arange(-15, -10)
            g_range = numpy.arange(-15, -10)
        else:
            c_range = numpy.arange(ranges[0][0], ranges[0][1])
            g_range = numpy.arange(ranges[1][0], ranges[1][1])

        number_iterations = c_range.size * g_range.size
        it_counter = 1

        if path_filename[0] is "":
            name = "svr_" + path_filename[1]
        else:
            name = path_filename[0] + "/svr_" + path_filename[1]
        best_param = BestParam(["c", "g"], (c_range, g_range),
                               name=name)

        print("[", end="")
        sys.stdout.flush()

        c_counter = 0
        for param_c in c_range:  # [-24, 25)
            g_counter = 0
            for param_g in g_range:  # [1, 4)
                cv_tr_error, cv_te_error = \
                    self.time_series_cross_validation(database,
                                                      params=[param_kernel_function,
                                                              2 ** param_c,
                                                              2 ** param_g,
                                                              0.1],
                                                      number_folds=10,
                                                      scale=scale)

                best_param.update(cv_te_error, [param_c, param_g],
                                  [c_counter, g_counter])

                g_counter += 1

                if (g_counter + c_counter*g_range.size)/number_iterations >= \
                                it_counter*0.1:
                    it_counter += 1
                    print(".", end="")
                    sys.stdout.flush()

            c_counter += 1

        print("]")

        if save:
            best_param.write_file(show=False)
            best_param.plot_error_surface(show=False, save=True)

        self.param_kernel_function = param_kernel_function
        self.param_c, self.param_g = best_param.get_best_param_rmse()
        self.param_c = 2 ** self.param_c
        self.param_g = 2 ** self.param_g

        return best_param
