# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

from .stock import Stock
from .regressors.regressor import Regressor
from elm.mltools import Error
from elm.mltools import DataProcess
from elm.mltools import split_sets
from elm import mltools


class RegressorStock:
    """
            Class that join Regressor and Stock classes.
    """

    def __init__(self, regressor, stock):

        if type(regressor) is Regressor:
            self.regressor = regressor

        elif type(regressor) is dict:
            if 'type' not in regressor or 'params' not in regressor:
                print("Error: Invalid constructor for Regressor.\n"
                      "regressor must be a Regressor type or a dictionary "
                      "containing 'type' and 'params' keys.")
                return
            else:
                self.regressor = Regressor(regressor_type=regressor['type'],
                                           regressor_params=regressor['params'])
        else:
            print("Error: Invalid constructor for Regressor.\n"
                      "regressor must be a Regressor type or a dictionary "
                      "containing 'type' and 'params' keys.")
            return

        if type(stock) is Stock:
            self.stock = stock

        elif type(stock) is dict:
            if 'name' not in stock or 'file_name' not in stock:
                print("Error: Invalid constructor for Stock.\n"
                      "stock must be a Stock type or a dictionary "
                      "containing 'name' and 'file_name' keys.")
                return
            else:
                self.stock = Stock(name=stock['name'],
                                   file_name=stock['file_name'])
        else:
            print("Error: Invalid constructor for Stock.\n"
                      "stock must be a Stock type or a dictionary "
                      "containing 'name' and 'file_name' keys.")
            return

        self.h_param = 1
        self.k_param = 1

        # DataProcess object
        self.dp = []

        # Last training, testing and predicting results
        self.tr_result = []
        self.te_result = []
        self.pr_result = []
        self.tri_result = []

        # To pickle
        self.stock_path = self.stock.name + "/"

        self.model_path = "regressors/models/"
        self.model_name = ""
        self.results_path = "data/"

    def check_regressor(self, series_type, window_size, scale, transf,
                        it_tr=False):
        """
            Check if a model exists and was already saved before.
        """

        self.model_name = self.regressor.type + "_" + series_type + "_ws_" + \
                          str(window_size) + "_sc_" + str(scale) + "_tr_" + \
                          str(transf) + ".rs"

        if it_tr is True:
            self.model_name += "it"

        if not os.path.exists(self.model_path + self.stock_path):
            os.makedirs(self.model_path + self.stock_path)

            return False
        else:

            file_name = self.model_path + self.stock_path + self.model_name
            if os.path.exists(file_name) or os.path.exists(file_name+"1"):
                return True
            else:
                return False

    def load_model(self):
        """
            Load a RegressorStock object to memory.

            saved model
        """

        file_name = self.model_path + self.stock_path + self.model_name

        try:
            with open(file_name, 'rb') as f:
                rs_model = pickle.load(f)
                self.__dict__.update(rs_model)

        except:
            raise Exception("Error while loading ", file_name)

        return self.regressor

    def save_model(self):
        """
            Save current RegressorStock object .
        """

        file_name = self.model_path + self.stock_path + self.model_name
        # self.regressor.save_model(file_name)

        try:
            with open(file_name, 'wb') as f:
                pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
                # pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        except:
            raise Exception("Error while saving ", file_name)

        else:
            print("Saved model as: ", file_name, "\n\n")

    def create_model3(self, series_type="return", window_size=10, scale=None,
                     transf=None):
        """

        """

        # Create data matrix with input and output patterns
        self.stock.create_database(series_type=series_type,
                                   window_size=window_size)

        data = self.stock.get_database()

        ########################
        #### Pre Processing ####
        ########################

        # 1 - Split data matrix in training and testing sets

        self.dp = DataProcess(scale_method=scale, transform_method=transf,
                              transform_params=None)
        # tr, te = self.dp.split_sets(data, n_test_samples=predict_horizon)
        tr, te = self.dp.split_sets(data, training_percent=.9)

        # 2 - Scale and Apply a transformation to the data
        train_set, test_set = self.dp.auto(tr, te)

        # 3- Check if there is a trained regressor with same parameters, if so,
        # load it to predict desired horizon, if don't, perform a search for
        # best parameters, train it and predict

        if self.check_regressor(series_type, window_size, scale, transf):
            self.regressor = self.load_model()

        else:

            # Starts a brute force search through all parameters range space
            # self.search_best_param(database=data, dataprocess=self.dp,
            #                        series_type=series_type, full_search=True,
            #                        save=True,
            #                        path_filename=(self.results_path +
            #                                       self.stock_path,
            #                                       self.model_name))

            self.search_opt(database=data, dataprocess=self.dp,
                            series_type=series_type, full_search=True,
                            save=True,
                            path_filename=(self.results_path +
                                           self.stock_path,
                                           self.model_name))

            # Use the best parameter found and train the regressor
            self.tr_result = self.train(training_matrix=train_set)

            # Save it !
            self.save_model()

        self.te_result = self.test(testing_matrix=test_set)
        # print(self.stock.name, " - ", series_type)
        # self.regressor.regressor.print_parameters()

    def create_model(self, series_type="return", window_size=4,
                     dataprocess=None, tr_percent=.8):
        """
            Search if there is an available model for this time series.

            if exists, load to memory.
            else performs a search for best hyper parameters, train model
            with time series and save to disk.

        """

        if dataprocess is None:
            self.dp = DataProcess()
        else:
            self.dp = dataprocess

        # Check if there is a regressor that performed a search before
        # for this series
        if self.check_regressor(series_type, window_size, self.dp.scale_method,
                                self.dp.scale_method):

            self.regressor = self.load_model()
            # self.print_parameters()

        else:

            # Create data matrix with input and output patterns
            self.create_database(window_size=window_size,
                                 series_type=series_type)

            data = self.stock.get_database()

            ########################
            #### Pre Processing ####
            ########################

            # 1 - Split data matrix in training and testing sets
            tr, te = mltools.split_sets(data, training_percent=tr_percent)

            # 2 - Scale and Apply a transformation to the data
            train_set, test_set = self.dp.auto(tr, te)

            # Search best hyper parameters and automatically set it
            params = self.search_param(database=data, dataprocess=self.dp,
                                       cv="ts", cv_nfolds=10, of="rmse",
                                       eval=50)

            # Use the best parameter found and train the regressor
            self.tr_result = self.train(train_set, params)
            self.te_result = self.test(test_set)
            self.te_result.print_errors()

            # Save it !
            self.save_model()

    def create_model_it(self, series_type="return", window_size=4,
                        dataprocess=None, sliding_window=168, k=1):
        """
            Search if there is an available model for this time series.

            if exists, load to memory.
            else performs a search for best hyper parameters, train model
            with time series and save to disk.

        """

        if dataprocess is None:
            self.dp = DataProcess()
        else:
            self.dp = dataprocess

        # Check if there is a regressor that performed a search before
        # for this series
        if self.check_regressor(series_type, window_size, self.dp.scale_method,
                                self.dp.transform_method, it_tr=True):
            self.load_model()

        else:

            # Create data matrix with input and output patterns
            self.create_database(window_size=window_size,
                                 series_type=series_type)

            data = self.stock.get_database()
            print(data.shape)

            # If performed a cross-validation before, use best parameters
            if self.regressor.get_cv_flag():
                params = self.regressor.get_cv_params()
            else:
                # Search best hyper parameters and automatically set it
                params = self.search_param(database=data, dataprocess=self.dp,
                                           cv="ts", cv_nfolds=10, of="rmse",
                                           eval=50)

            # self.tri_result = self.train_it(data, sliding_window=sliding_window,
            #                                 k=k, params=params, search=True)

            # print("Iterative training results:")
            # self.tri_result.print_errors()
            # self.tri_result.print_values()

            # Save it !
            self.save_model()

    def test_model_it(self, dataprocess=None, series_type="return",
                      window_size=4, sliding_window=168, k=1,
                      trading_samples=142):

        if dataprocess is None:
            dataprocess = DataProcess()

        # Create data matrix with input and output patterns
        self.create_database(window_size=window_size,
                             series_type=series_type)

        data = self.stock.get_database()

        # If performed a cross-validation before, use best parameters
        if self.regressor.get_cv_flag():
            params = self.regressor.get_cv_params()
        else:
            # Search best hyper parameters and automatically set it
            params = self.search_param(database=data, dataprocess=dataprocess,
                                       cv="ts", cv_nfolds=10, of="rmse",
                                       eval=50)

        estimate_param = data.shape[0] - trading_samples
        estimate_data = data[0:estimate_param, :]
        tri_result = self.train_it(data,
                                        sliding_window=sliding_window,
                                        k=k, params=params, search=False)

        errors = tri_result.get_error()
        # print(estimate_result.predicted_targets.shape)
        # estimate_result.print_errors()





    def auto(self, series_type="return", predict_horizon=5, window_size=4,
             scale=None, scale_output=False, transf=None):
        """

        """

        # Create data matrix with input and output patterns
        self.stock.create_database(series_type=series_type,
                                   window_size=window_size)

        data = self.stock.get_database()

        ########################
        #### Pre Processing ####
        ########################

        # 1 - Split data matrix in training and testing sets

        self.dp = DataProcess(scale_method=scale, scale_output=scale_output,
                              transform_method=transf,
                              transform_params=None)

        tr, te = self.dp.split_sets(data, n_test_samples=predict_horizon)

        # 2 - Scale and Apply a transformation to the data
        train_set, test_set = self.dp.auto(tr, te)

        # 3- Check if there is a trained regressor with same parameters, if so,
        # load it to predict desired horizon, if don't, perform a search for
        # best parameters, train it and predict

        if self.check_regressor(series_type, window_size, scale, transf):
            self.regressor = self.load_model()

        else:

            # Starts a brute force search through all parameters range space
            # self.search_best_param(database=data, dataprocess=self.dp,
            #                        series_type=series_type, full_search=True,
            #                        save=True,
            #                        path_filename=(self.results_path +
            #                                       self.stock_path,
            #                                       self.model_name))

            self.search_opt(database=data, dataprocess=self.dp,
                            series_type=series_type, full_search=True,
                            save=True,
                            path_filename=(self.results_path +
                                           self.stock_path,
                                           self.model_name))

            # # Use the best parameter found and train the regressor
            self.tr_result = self.train(training_matrix=train_set)
            self.te_result = self.test(testing_matrix=test_set)

            # Save it !
            self.save_model(self.regressor)

        # print("Iterative training of ", self.stock.name)
        # self.tri_result = self.train_iterative(data, sw=168, k=1)
        # self.tri_result.print_errors()
        # # self.regressor.regressor.print_parameters()
        #
        # self.tr_result = self.tri_result
        # self.te_result = self.tri_result
        # self.pr_result = self.tri_result
        # self.regressor.regressor.print_parameters()

        # Performs a test and prediction
        self.te_result = self.test(testing_matrix=test_set)
        predicted_targets = self.predict(horizon=predict_horizon)

        self.pr_result = Error(expected=self.te_result.expected_targets,
                               predicted=predicted_targets)
        #
        # ########################
        # ####### Printing #######
        # ########################
        #
        # # print("Training Errors:")
        # # tr_result.print_errors()
        #
        # # print("Testing Errors:")
        # # self.te_result.print_errors()
        # #
        # # print("Prediction Errors:")
        # # self.pr_result.print_errors()
        # # pr_result.print_values()

        return self.tr_result, self.te_result, self.pr_result

    def minimize_database(self):
        """

            Args:
                function_type -

            Returns:
                best_param

        """

        number_test_samples = 60

        self.stock.create_database(window_size=30)
        original_database = self.stock.data_matrix

        self.regressor.set_database(original_database,
                                    number_test_samples=number_test_samples)
        # tr, te = self.elm.time_series_cross_validation(number_folds=10)
        tr, te = self.regressor.time_series_cross_validation(number_folds=10)

        best_te = np.mean(te)

        database = original_database
        attempt = 0
        sizes = []

        print("initial size: ", database.shape[0], " rmse: ", best_te)
        sizes.append(database.shape[0])

        while True:
            i = int(database.shape[0] / 2)
            database = database[i:, :]

            sizes.append(database.shape[0])

            self.regressor.set_database(database,
                                        number_test_samples=number_test_samples)
            tr, te = self.regressor.time_series_cross_validation(
                number_folds=10)
            te = np.mean(te)

            print("size: ", database.shape[0], " rmse: ", te)

            if te < best_te:
                attempt = 0

            else:
                attempt += 1
                if attempt == 2:
                    break

        print(sizes)
        print("final size: ", sizes[-3])

    ## Stock Methods

    def plot_stock(self, show=True, save=False):
        self.stock.plot_stock(show=show, save=save)

    def plot_return(self, show=True, save=False):
        self.stock.plot_return(show=show, save=save)

    def create_database(self, window_size=30, k=1, series_type="return"):
        self.stock.create_database(window_size=window_size, k=k, series_type=series_type)

    def get_stock_name(self):
        return self.stock.get_stock_name()

    def get_return_series(self):
        return self.stock.get_return_series()

    def get_time_series(self):
        return self.stock.get_time_series()

    # Regressor methods

    @mltools.copy_doc_of(Regressor.train)
    def train(self, training_matrix, params=[]):
        return self.regressor.train(training_matrix, params)

    @mltools.copy_doc_of(Regressor.train_it)
    def train_it(self, database_matrix, params=[], dataprocess=None,
                 sliding_window=168, k=1, search=False):
        return self.regressor.train_it(database_matrix, params, dataprocess,
                                       sliding_window, k, search)

    @mltools.copy_doc_of(Regressor.predict_it)
    def predict_it(self, horizon=1, dataprocess=None):
        return self.regressor.predict_it(horizon, dataprocess)

    @mltools.copy_doc_of(Regressor.test)
    def test(self, testing_matrix, predicting=False):
        return self.regressor.test(testing_matrix, predicting)

    @mltools.copy_doc_of(Regressor.predict)
    def predict(self, horizon=30):
        return self.regressor.predict(horizon)

    @mltools.copy_doc_of(Regressor.print_parameters)
    def print_parameters(self):
        self.regressor.print_parameters()

    @mltools.copy_doc_of(Regressor.search_param)
    def search_param(self, database, dataprocess=None, path_filename=("", ""),
                     save=False, cv="ts", cv_nfolds=10, of="rmse", kf=None,
                     f=None, opt_f="cma-es", eval=50, print_log=True):

        return self.regressor.search_param(database, dataprocess,
                                           path_filename, save, cv,
                                           cv_nfolds, of, kf, f, opt_f,
                                           eval, print_log)

    # def s_expected_return_mean(self, samples=0, show=False):
    #     self.stock.expected_return_mean(samples=samples, show=show)
    #
    # def s_risk_std(self, samples=0, show=False):
    #     self.stock.risk_std(samples=samples, show=show)
    #
    # def s_calculate_return(self):
    #     self.stock.calculate_return()
    #
    # def s_get_data(self, file_name, reverse=True):
    #     self.stock.get_data(file_name=file_name, reverse=reverse)
    #