# -*- coding: utf-8 -*-

# Python2 support
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import numpy as np
import sys
import pickle

from elm.mltools import MLTools
from elm.mltools import Error
from elm.mltools import copy_doc_of


try:
    import bob.learn.mlp
except ImportError:
    _BOB_MLP_AVAILABLE = 0
else:
    _BOB_MLP_AVAILABLE = 1

"""
    To install bob.learn.mlp on ubuntu/mint system

    1- sudo apt-get install
    cmake
    libboost1.55-dev libboost-python1.55-dev libboost-system1.55-dev
    libboost-iostreams1.55-dev libboost-filesystem1.55-dev
    libblitz-doc libblitz0-dev libblitz0ldbl
    libhdf5-dev

    2- sudo pip install bob.extension
    3- sudo pip install bob.blitz
    4- sudo pip install bob.core
    5- sudo pip install bob.io.base
    6- sudo pip install bob.math
    7- sudo pip install bob.learn.activation
    8- sudo pip install bob.learn.mlp

"""

"""
    To install FANN

    1- Download source code, go to FANN directory and "cmake ." then "ldconfig"
    2- Test by "make examples/runtest"
    3- sudo apt-get install swig2.0
    4- install from https://github.com/troiganto/fann2

"""

# TODO
# Memory leak associated with MLP training, remove mlp initialization at
# train beginning



class MLP(MLTools):
    """
        MLP
    """

    def __init__(self, params=[]):
        super(self.__class__, self).__init__()

        if not _BOB_MLP_AVAILABLE:
            raise Exception("Please install 'bob.learn.mlp' package to use MLP."
                            "Search in portfolio-ai doc to references.")

        self.regressor_name = "mlp"

        self.available_training_function = ["backpropagation"]
        self.available_activation_function = ["sigmoid"]

        # Default parameters
        self.default_param_hidden_neurons = (16, 4)
        self.default_param_hidden_neurons_activation_function = "sigmoid"
        self.default_param_output_neurons_activation_function = "sigmoid"
        self.default_param_training_function = "backpropagation"
        self.default_param_cost_function = "squareerror"
        self.default_param_learning_rate = 0.009
        self.default_param_momentum = 0.95
        self.default_param_epochs = 5000

        # Initialized parameters values
        if not params:
            self.param_hidden_neurons = self.default_param_hidden_neurons
            self.param_hidden_neurons_activation_function = \
                self.default_param_hidden_neurons_activation_function
            self.param_output_neurons_activation_function = \
                self.default_param_output_neurons_activation_function
            self.param_training_function = self.default_param_training_function
            self.param_cost_function = self.default_param_cost_function
            self.param_learning_rate = self.default_param_learning_rate
            self.param_momentum = self.default_param_momentum
            self.param_epochs = self.default_param_epochs
        else:
            self.param_hidden_neurons = params[0]
            self.param_hidden_neurons_activation_function = params[1]
            self.param_output_neurons_activation_function = params[2]
            self.param_training_function = params[3]
            self.param_cost_function = params[4]
            self.param_learning_rate = params[5]
            self.param_momentum = params[6]
            self.param_epochs = params[7]

        self.mlp = "MLP not trained"
        self.trainer = "MLP not trained"
        self.architecture = "MLP not trained"

    # ########################
    # Private Methods
    # ########################

    # def _local_train_fann(self, training_patterns, training_expected_targets,
    #                       params):
    #     """
    #         MLP training method definided by Fred 09 paper.
    #     """
    #
    #     # If params not provided, uses initialized parameters values
    #     if not params:
    #         pass
    #     else:
    #         self.param_hidden_neurons = params[0]
    #         self.param_hidden_neurons_activation_function = params[1]
    #         self.param_output_neurons_activation_function = params[2]
    #         self.param_training_function = params[3]
    #         self.param_cost_function = params[4]
    #         self.param_learning_rate = params[5]
    #         self.param_momentum = params[6]
    #         self.param_epochs = params[7]
    #
    #     training_expected_targets = training_expected_targets.reshape(-1, 1)
    #
    #     input_neurons = training_patterns.shape[1]
    #     output_neurons = training_expected_targets.shape[1]
    #     self.architecture = \
    #         (input_neurons, ) + self.param_hidden_neurons + (output_neurons, )
    #
    #     # Create a instance of a MLP object
    #     self.mlp = libfann.neural_net()
    #     self.mlp.create_standard_array(list(self.architecture))
    #
    #     # Define activation functions
    #     if self.param_hidden_neurons_activation_function == "sigmoid":
    #         self.mlp.set_activation_function_hidden(libfann.SIGMOID)
    #     else:
    #         print("Error: Invalid hidden activation function.")
    #         return
    #
    #     if self.param_output_neurons_activation_function == "sigmoid":
    #         self.mlp.set_activation_function_output(libfann.SIGMOID)
    #     else:
    #         print("Error: Invalid output activation function.")
    #         return
    #
    #     self.mlp.set_training_algorithm(libfann.TRAIN_BATCH)
    #     self.mlp.set_learning_rate(self.param_learning_rate)
    #     self.mlp.set_learning_momentum(self.param_momentum)
    #
    #     self.mlp.print_parameters()
    #
    #     # Special training
    #     # Defined on paper FRED 09
    #     n_training_epochs = 1000
    #     n_validation_epochs = 10
    #     n_iterations = int(self.param_epochs / n_training_epochs)
    #
    #     # Validation set parameters
    #     vset_percentage = 4
    #     vset_size = \
    #         int(np.ceil(training_patterns.shape[0]*vset_percentage/100))
    #
    #     va_patterns = training_patterns[-vset_size:, :]
    #     va_expected_targets = training_expected_targets[-vset_size:, :]
    #     va_best_rmse = 9999.9
    #
    #     tr_patterns = training_patterns[0:-vset_size, :]
    #     tr_expected_targets = training_expected_targets[0:-vset_size, :]
    #
    #     sub_it_data = libfann.training_data()
    #     sub_it_data.set_train_data(tr_patterns,
    #                                tr_expected_targets)
    #
    #     va_it_data = libfann.training_data()
    #     va_it_data.set_train_data(training_patterns,
    #                               training_expected_targets)
    #
    #     best_training_weights = []
    #     best_training_biases = []
    #     best_ann = []
    #     for it in range(n_iterations):
    #
    #         # Perform a sub iteration with only training_patterns'
    #         for epoch in range(n_training_epochs):
    #             self.mlp.train_epoch(sub_it_data)
    #
    #         # Add validation set to training
    #         for epoch in range(n_validation_epochs):
    #             self.mlp.train_epoch(va_it_data)
    #
    #         # Predict values from validation set and calculate RMSE error
    #         va_predicted_targets = []
    #         for i in range(va_patterns.shape[0]):
    #             va_predicted_targets.append(self.mlp.run(va_patterns[i, :]))
    #
    #         va_errors = Error(va_expected_targets, va_predicted_targets)
    #
    #         # Check if this error was the best one, if so, save current
    #         # weights and biases
    #         if va_errors.get_rmse() < va_best_rmse:
    #             va_best_rmse = va_errors.get_rmse()
    #             best_training_weights = self.mlp.get_connection_array()
    #
    #         print(va_best_rmse)
    #
    #     # Apply best weights and biases
    #     self.mlp.set_weight_array(best_training_weights)
    #
    #     # Calculates training errors
    #     tr_predicted_targets = []
    #     for i in range(training_patterns.shape[0]):
    #         tr_predicted_targets.append(self.mlp.run(training_patterns[i, :]))
    #
    #     return tr_predicted_targets

    def _local_train(self, training_patterns, training_expected_targets,
                     params):
        """
            MLP training method definided by Fred 09 paper.
        """

        # If params not provided, uses initialized parameters values
        if not params:
            pass
        else:
            self.param_hidden_neurons = params[0]
            self.param_hidden_neurons_activation_function = params[1]
            self.param_output_neurons_activation_function = params[2]
            self.param_training_function = params[3]
            self.param_cost_function = params[4]
            self.param_learning_rate = params[5]
            self.param_momentum = params[6]
            self.param_epochs = params[7]

        training_expected_targets = training_expected_targets.reshape(-1, 1)

        input_neurons = training_patterns.shape[1]
        output_neurons = training_expected_targets.shape[1]
        self.architecture = \
            (input_neurons, ) + self.param_hidden_neurons + (output_neurons, )

        # Create a instance of a MLP object
        self.mlp = bob.learn.mlp.Machine(self.architecture)

        # Define activation functions
        if self.param_hidden_neurons_activation_function == "sigmoid":
            self.mlp.hidden_activation = bob.learn.activation.Logistic()
        else:
            raise Exception("Error: Invalid hidden activation function.")

        if self.param_output_neurons_activation_function == "sigmoid":
            self.mlp.output_activation = bob.learn.activation.Logistic()
        else:
            raise Exception("Error: Invalid output activation function.")

        # Create a cost function object
        if self.param_cost_function == "squareerror":
            self.cost_f = bob.learn.mlp.SquareError(self.mlp.output_activation)
        else:
            raise Exception("Error: Invalid cost function.")

        # Create a trainer
        if self.param_training_function == "backpropagation":

            #  Creates a backpropagation trainer
            self.trainer = bob.learn.mlp.BackProp(self.param_epochs,
                                                  self.cost_f,
                                                  self.mlp,
                                                  train_biases=False)
        else:
            raise Exception("Error: Invalid training function.")

        self.trainer.momentum = self.param_momentum
        self.trainer.learning_rate = self.param_learning_rate

        # Initialize trainer
        self.trainer.initialize(self.mlp)

        # Special training
        # Defined on paper FRED 09
        n_training_epochs = 1000
        n_validation_epochs = 10
        n_iterations = int(self.param_epochs / n_training_epochs)

        # Validation set parameters
        vset_percentage = 4
        vset_size = int(np.ceil(training_patterns.shape[0]*vset_percentage/100))

        va_patterns = training_patterns[-vset_size:, :]
        va_expected_targets = training_expected_targets[-vset_size:, :]
        va_best_rmse = 9999.9

        tr_patterns = training_patterns[0:-vset_size, :]
        tr_expected_targets = training_expected_targets[0:-vset_size, :]

        best_training_weights = []
        best_training_biases = []
        for it in range(n_iterations):
            # print("iteration: ", it+1)

            # Perform a sub iteration with only training_patterns'
            self.trainer.batch_size = tr_patterns.shape[0]
            for epoch in range(n_training_epochs):
                self.trainer.train(self.mlp,
                                   tr_patterns,
                                   tr_expected_targets)

            # Add validation set to training
            self.trainer.batch_size = training_patterns.shape[0]
            for epoch in range(n_validation_epochs):
                self.trainer.train(self.mlp,
                                   training_patterns,
                                   training_expected_targets)

            # Predict values from validation set and calculate RMSE error
            va_predicted_targets = self.mlp(va_patterns)
            va_errors = Error(va_expected_targets, va_predicted_targets)

            # Check if this error was the best one, if so, save current
            # weights and biases

            if va_errors.get_rmse() < va_best_rmse:
                va_best_rmse = va_errors.get_rmse()
                best_training_weights = self.mlp.weights
                best_training_biases = self.mlp.biases

            print("it: ", it, va_errors.get_rmse())

        # Apply best weights and biases
        self.mlp.weights = best_training_weights
        self.mlp.biases = best_training_biases

        # Calculates training errors
        training_predicted_targets = self.mlp(training_patterns)

        return training_predicted_targets

    def _local_test(self, testing_patterns, testing_expected_targets,
                    predicting):

        testing_predicted_targets = self.mlp(testing_patterns)

        return testing_predicted_targets

    # ########################
    # Public Methods
    # ########################

    def print_parameters(self):
        """
            Print parameters values.
        """

        print()
        print("Regressor Parameters")
        print()
        print("Network architecture: ", self.architecture)
        print("Hidden neurons activation function: ",
              self.param_hidden_neurons_activation_function)
        print("Output neurons activation function: ",
              self.param_output_neurons_activation_function)
        print("Training function: ", self.param_training_function)
        print("Cost function: ", self.param_cost_function)
        print("Learning rate: ", self.param_learning_rate)
        print("Momentum (Inertia): ", self.param_momentum)
        print("Epochs: ", self.param_epochs)
        self.print_cv_log()

    # def save_model(self, file_name):
    #
    #     try:
    #         # First save all class attributes
    #
    #         file = file_name + "1"
    #         with open(file, 'wb') as f:
    #             pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #         # Now save bob MLP
    #
    #         import bob.io.base
    #
    #         file = file_name + "2"
    #         file = bob.io.base.HDF5File(file, mode="w")
    #         self.mlp.save(file)
    #
    #     except:
    #         raise Exception("Error while saving ", file_name)
    #
    #     else:
    #         print("Saved model as: ", file_name, "\n\n")
    #
    # def load_model(self, file_name):
    #
    #     try:
    #         # First load all class attributes
    #
    #         file = file_name + "1"
    #         with open(file, 'rb') as f:
    #             self = pickle.load(f)
    #
    #         import bob.io.base
    #
    #         # Temporary network
    #         self.mlp = bob.learn.mlp.Machine(self.architecture)
    #
    #         file = file_name + "2"
    #         file = bob.io.base.HDF5File(file, mode="r")
    #         self.mlp.load(file)
    #     except:
    #         raise Exception("Error while loading ", file_name)
    #
    #     return self

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

    def search_param(self, database, dataprocess=None, path_filename=("", ""),
                     save=False, cv="ts", cv_nfolds=10, of="rmse", kf=None,
                     opt_f="cma-es", eval=50, print_log=True):
        """
            TODO


            See Also:
                http://optunity.readthedocs.org/en/latest/user/index.html
        """

        # TODO


        params = [self.param_hidden_neurons,
                  self.param_hidden_neurons_activation_function,
                  self.param_output_neurons_activation_function,
                  self.param_training_function,
                  self.param_cost_function,
                  self.param_learning_rate,
                  self.param_momentum,
                  self.param_epochs]

        # MLTools Attribute
        # self.has_cv = True
        # self.cv_name = cv
        # self.cv_error_name = of
        # self.cv_best_error = best_function_error
        # self.cv_best_params = params

        return params