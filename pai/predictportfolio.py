# -*- coding: utf-8 -*-

"""
    This file contains Portofolio class and developed methods.
"""

from .stock import Stock
from .stock import ts_to_matrix
from .regressors.regressor import Regressor
from .regressorstock import RegressorStock

import os
import numpy as np
import datetime
from .portfolio import PortfolioBase
from .portfolio import get_data
from .portfolio import solve_sharpe
from .portfolio import solve_cvxopt
from .portfolio import init_error
from .portfolio import pareto
from .portfolio import calc_portfolio_metrics
from .portfolio import pareto_weights
from .portfolio import gof_estimate_data
from .portfolio import calc_ar
from .portfolio import calc_pcm
from .portfolio import calc_ti
from .portfolio import plot_ar

from elm import mltools

try:
    from scipy import optimize
except ImportError:
    _SCIPY_AVAILABLE = 0
else:
    _SCIPY_AVAILABLE = 1

try:
    from cvxopt import matrix, solvers
except ImportError:
    _CVXOPT_AVAILABLE = 0
else:
    _CVXOPT_AVAILABLE = 1

try:
    import optunity
except ImportError:
    _OPTUNITY_AVAILABLE = 0
else:
    _OPTUNITY_AVAILABLE = 1

try:
    import pandas.io.data as web
    from pandas.tseries.offsets import Day
except ImportError:
    _PANDAS_AVAILABLE = 0
else:
    _PANDAS_AVAILABLE = 1

try:
    from prettytable import PrettyTable
except ImportError:
    _PRETTYTABLE_AVAILABLE = 0
else:
    _PRETTYTABLE_AVAILABLE = 1

try:
    from matplotlib import pyplot as plt
except ImportError:
    _MATPLOT_AVAILABLE = 0
else:
    _MATPLOT_AVAILABLE = 1


class Predict(PortfolioBase):
    """
            Portfolio class
    """

    def __init__(self, regressorstock=None):
        super(self.__class__, self).__init__()

        if type(regressorstock) is list:
            if type(regressorstock[0]) is not RegressorStock:
                init_error()
                return

            self.portfolio = regressorstock

        elif type(regressorstock) is RegressorStock:
            self.portfolio = [regressorstock]

        elif regressorstock is None:
            self.portfolio = []

        else:
            init_error()

        # Weight from each asset on the portfolio
        self.weights = []

        # Weights from max Sharpe ratio point
        self.max_sharpe_weights = []

        # Expected returns from each asset on the portfolio
        self.er = []

        # Covariance matrix from all assets on the portfolio
        self.cov = []

    def _exist(self, stock_name):
        """
            Find for stock in portfolio, if exists return a pointer to it,
            if dont returns False.

        """

        for s in self.portfolio:
            if s.stock.name == stock_name:
                return s
        return False

    def __print_results_sharpe(self):
        """
            Print the assets proportion, portfolio return and risk for the
            max sharpe value.
        """

        # Print results
        print("\n\nOptimization results: ")
        for sr in range(len(self.portfolio)):
            print("Asset: ", self.portfolio[sr].stock.name,
                  np.around(100 * self.weights[sr]/np.sum(self.weights),
                            decimals=2), "%")

        print("\n\nPortfolio")
        print("Return - ", np.dot(self.weights, self.er))
        print("Risk - ", np.dot(np.dot(self.weights, self.cov), self.weights.T))

    def __get_cov(self, risk, trading_samples=None):
        """
            Calculates covariance matrix from assets, if risk is equal to
            "returns" than the used variables will be a series of returns
            from each asset, if risk is equal to "errors" than the used
            variables will be a series of testing errors from training phase
            of regressor.
        """

        # First, look for the minimum length, it will be the size of
        # our matrix

        ret_size = 9999999999
        for sr in self.portfolio:

            # If risk measurement is based upon returns is needed to find the
            # minimum length of the return array in each asset,
            # so is possible to create a matrix with equal columns size
            # If risk measurement is based upon errors is needed to do the
            # same thing because the errors array is based upon the test size
            # used for training and this can be different for regressors

            if risk == "returns":
                if sr.stock.returns.size < ret_size:
                    ret_size = sr.stock.returns.size
            else:
                if sr.tri_result.expected_targets.size < ret_size:
                    ret_size = sr.tri_result.expected_targets.size
            # print("retsize: ", ret_size)

        ret_matrix = np.empty((len(self.portfolio), ret_size))

        for sr in range(len(self.portfolio)):
            if risk == "returns":
                ret_matrix[sr, :] = self.portfolio[sr].stock.returns[-ret_size:]
            else:
                ret_matrix[sr, :] = \
                    self.portfolio[sr].tri_result.get_error()[-ret_size:]

        cov = np.cov(ret_matrix)
        # a = np.linalg.cholesky(cov)
        # print(a)

        return cov

    def add(self, stock_name, regressor_name="elmk", window_size=4,
            dataprocess=None):
        """
            Add a stock to portofolio.
        """

        if type(stock_name) is not str:
            raise Exception("stock_name must be a 'str' asset name from "
                            "BOVESPA.")

        if self._exist(stock_name):
            print("Warning: Couldn't add ", stock_name, " to portfolio ",
                  "because stock already exists.")
            return

        # Create a temporary stock object
        stock = get_data(stock_name)
        # stock.create_database(window_size=window_size, series_type="return")

        # Create a temporary regressor object
        regressor = Regressor(regressor_type=regressor_name)

        # Create a regressorstock object and add it to the portfolio list
        regressor = RegressorStock(regressor=regressor, stock=stock)

        regressor.create_model_it("return", window_size,
                                  dataprocess=dataprocess)

        self.portfolio.append(regressor)

    def list(self):
        """
            Print a table with all assets.
        """

        if _PRETTYTABLE_AVAILABLE:

            table = PrettyTable(["Stock name",
                                 "Regressor name",
                                 "Expected return",
                                 "Risk (testing error variance)",
                                 "Regressor Cross-Validation RMSE",
                                 "Regressor training IT test RMSE"])
            table.align["Stock name"] = "l"

            print("Listing portfolio: \n")
            for sr in self.portfolio:
                er = sr.predict_it(horizon=1)
                tri_std = sr.tri_result.get_std() ** 2
                tri_rmse = sr.tri_result.get("rmse")
                cv_rmse = sr.regressor.regressor.cv_best_error

                table.add_row([sr.stock.name, sr.regressor.type, er[0],
                               tri_std, cv_rmse, tri_rmse])

            print()
            print(table.get_string(sortby="Stock name"))
            print()

        else:
            print("For better table format install 'prettytable' package.")

            print("Listing portfolio: \n")
            print()
            print("Stock name | ", "Regressor name | ", "Expected return | ",
                  "Risk (testing error variance) | ",
                  "Regressor Cross-Validation RMSE | ",
                  "Regressor training IT test RMSE")

            for sr in self.portfolio:
                er = sr.predict_it(horizon=1)
                tri_std = sr.tri_result.get_std() ** 2
                tri_rmse = sr.tri_result.get("rmse")
                cv_rmse = sr.regressor.regressor.cv_best_error

                print(sr.stock.name, sr.regressor.type, er[0], tri_std,
                      cv_rmse, tri_rmse)
            print()

    def __print_weights_sharpe(self, we):

        if _PRETTYTABLE_AVAILABLE:
            table = PrettyTable(["Stock name", "%"])
            table.align["Stock name"] = "l"

            print("Max Sharpe portfolio")
            for i, sr in enumerate(self.portfolio):
                table.add_row([sr.stock.name,
                               np.around(100 * (we[i]/np.sum(we)), decimals=2)
                              ])

            print(table.get_string(sortby="Stock name"))
            print()

        else:
            print("Max Sharpe portfolio")
            print()
            print("Stock name | ", "%")
            for i, sr in enumerate(self.portfolio):
                print(sr.stock.name,
                      np.around(100 * (we[i]/np.sum(we)), decimals=2))

            print()

    def optimize(self, risk="errors"):
        """
            Optimize the weights from current portfolio, find the max Sharpe
            ratio weight, a pareto of portfolios and plot it.
        """

        names = []
        self.er = []

        # Calculate expected return
        for sr in self.portfolio:
            names.append(sr.stock.name)
            self.er.append(sr.predict_it(horizon=1)[0])

        # Calculate covariance (risk) between assets from portfolio
        self.cov = self.__get_cov(risk)

        # Weekly risk-free rate (%)
        rf = 0.1193 / 100

        weights = []
        if len(self.portfolio) > 1:
            # Solve weights by maximizing sharpe ratio
            self.max_sharpe_weights = solve_sharpe(self.er, self.cov, rf)

            # Solve quadratic problem defined in Fred 09 paper
            for erd in np.linspace(np.min(self.er), np.max(self.er), 50):
                weights.append(solve_cvxopt(self.er, self.cov, erd))
        else:
            self.max_sharpe_weights = np.ones((1, 1))
            weights.append(np.ones((1, 1)))

        # Plot pareto
        assets = {"name": names, "er": self.er, "cov": self.cov}
        self._pareto(weights, assets)

        self.__print_weights_sharpe(self.max_sharpe_weights)

        p_er = np.dot(self.max_sharpe_weights, self.er)
        p_risk = np.dot(np.dot(self.max_sharpe_weights, self.cov), self.max_sharpe_weights)
        print("Portfolio Return: ", p_er)
        print("Portfolio Risk: ", p_risk)
        # self.__print_results()

    def test(self, risk_free=0.1193 / 100, trading_samples=142,
             estimating_samples=103):

        # trading_samples = 10
        # estimating_samples = 5
        # self.portfolio[0].tri_result.predicted_targets = np.arange(18, 34)
        # self.portfolio[0].tri_result.expected_targets = np.zeros((16,))

        pr_error = np.empty((len(self.portfolio), estimating_samples))

        data = []
        ref_data = []
        for t in range(trading_samples):

            er = []
            ref_er = []
            for rs in range(len(self.portfolio)):
                jump = self.portfolio[rs].tri_result.predicted_targets.size -\
                       trading_samples - estimating_samples

                if jump < 0:
                    print("Error: Not enough iterative training points, "
                          "set a lower  'k', a lower 'sliding_window' to "
                          "create more training data points, or set a lower "
                          "'trading_samples' or 'estimating_samples'.")
                    raise Exception("Error: Not enough data.")

                # # print()
                # tmp = self.portfolio[rs].tri_result.get_error()\
                #     [jump + t:jump + t + estimating_samples]
                # print(jump, "[", jump + t, " : ",
                #       jump+t+estimating_samples-1, "]",
                #       tmp)

                pr_error[rs, :] = self.portfolio[rs].tri_result.get_error()\
                    [jump + t:jump + t + estimating_samples]

                er.append(self.portfolio[rs].tri_result.
                    predicted_targets[jump + t + estimating_samples])
                ref_er.append(self.portfolio[rs].tri_result.
                    expected_targets[jump + t + estimating_samples])

            cov = np.cov(pr_error)

            sharpe_weights = solve_sharpe(er, cov, risk_free)
            ref_sharpe_weights = solve_sharpe(ref_er, cov, risk_free)
            # print(er, sharpe_weights)

            data.append({"er": er, "weights": sharpe_weights})
            ref_data.append({"er": ref_er, "weights": ref_sharpe_weights})

        #Calculate performance metrics
        ar, pcm, ti = self.calc_portfolio_metrics(data)

        ref_ar, ref_pcm, ref_ti = self.calc_portfolio_metrics(ref_data)

        print()
        print("Reference accumulated return: ", ref_ar)
        print("Reference PCM: ", ref_pcm)
        print("Reference TI: ", ref_ti)
        print()
        print("Accumulated return: ", ar)
        print("PCM: ", pcm)
        print("TI: ", ti)
        print()

    def _estimate(self, rs, ts, z, p, dataprocess, search, not_z=False):

        # print()
        # print("_estimate\n")

        if len(ts) < p + 2:
            print("Error: Not enough data for training/testing.")
            print("Increase sliding_window size or decrease p value.")
            exit()

        datamatrix = ts_to_matrix(ts, p)
        z_matrix = z[:datamatrix.shape[0]].reshape(-1, 1) * \
                   np.ones((1, datamatrix.shape[1]))

        if not_z:
            z_matrix *= 0

        data = datamatrix - z_matrix

        tr_data = data[:-1, :]
        te_data = data[-1, :].reshape(1, -1)
        # print("te_data")
        # print(te_data)
        # print()

        if dataprocess is not None:
            # Pre-process data
            tr_data, te_data = dataprocess.auto(tr_data, te_data)

        params = []
        if search:
            cv_data = np.vstack((tr_data, te_data))
            params = rs.search_param(database=cv_data,
                                     dataprocess=dataprocess,
                                     cv="ts", cv_nfolds=3, of="rmse",
                                     eval=50, print_log=False)

        # Train sliding window dataset
        rs.train(tr_data, params)

        # Predicted target with training_data - z_i ( r_t+1  )
        pr_t = rs.test(te_data)
        # print("pr_t")
        # pr_t.print_errors()
        # print()
        pr_t = pr_t.predicted_targets

        if dataprocess is not None:
            # Pos-process data
            pr_t = dataprocess.reverse_scale_target(pr_t)

        # Sum previous subtracted z_i value
        pr_t = pr_t.flatten()[0] + z_matrix[-1, 0]

        # print()
        return pr_t

    def benchmark(self, sw, p, k, estimate_size, trading_size, dataprocess=None,
                  search=False, risk_free=0.1193 / 100, not_z=False,
                  weights="sharpe"):

        # self.portfolio[0].stock.returns = np.arange(1, 51)

        def get_valid_portfolio(t, estimate_data):

            valid_portfolio = []

            for rs in self.portfolio:
                name = rs.get_stock_name()
                ts = rs.get_return_series()

                start = t
                end = start + estimate_size

                error = estimate_data[name]["er"][start:end] - \
                        estimate_data[name]["rr"][start:end]

                result = gof_estimate_data(error)
                if result["result"]:
                    valid_portfolio.append(rs)
                else:
                    print(t, "Removed ", name)

            return valid_portfolio

        if weights == "max_return":
            w_type = "er_max_return_weights"

        elif weights == "min_risk":
            w_type = "er_min_risk_weights"

        elif weights == "sharpe":
            w_type = "er_sharpe_weights"

        else:
            print("Invalid weight type.")
            return

        def wrapper_estimate_sw(rs, ts, offset=0, not_z=True):
            start = len(ts) - trading_size - estimate_size - (sw-1) + offset
            end = start + sw

            z = ts[start-k:end-k]
            ts = ts[start:end]

            er = self._estimate(rs, ts, z, p, dataprocess, search, not_z=not_z)
            rr = ts[-1]

            return er, rr

        def wrapper_estimate_es(t, rs, estimate_data):
            name = rs.get_stock_name()

            start = t
            end = start + estimate_size

            end = len(estimate_data[name]["er"])-1
            start = end - estimate_size

            # print("start: ", start, "end: ", end)

            er = estimate_data[name]["er"][end]
            rr = estimate_data[name]["rr"][end]

            error = estimate_data[name]["er"][start:end] - \
                    estimate_data[name]["rr"][start:end]

            # print(len(error))

            tocov = error
            return er, rr, tocov

        estimate_data = {}

        # Get estimate data used for expected returns and covariance
        for rs in self.portfolio:
            ts = rs.get_return_series()
            if len(ts) < trading_size + estimate_size + (sw-1) + k:
                print(len(ts), trading_size + estimate_size + (sw-1) + k)
                print("Error: Change parameters, not enough data to trade.")
                return

            name = rs.get_stock_name()

            n = estimate_size
            estimate_data[name] = {}
            estimate_data[name]["er"] = np.zeros(n)
            estimate_data[name]["rr"] = np.zeros(n)

            for e in range(estimate_size):
                er, rr = wrapper_estimate_sw(rs, ts, offset=e)
                estimate_data[name]["er"][e] = er
                estimate_data[name]["rr"][e] = rr

        import pprint

        print("")
        print("Trading")
        print("")
        trading_data = {}
        total_er = []
        total_rr = []
        for t in range(trading_size):
            trading_data[t] = {}

            valid_portfolio = get_valid_portfolio(t, estimate_data)

            n = len(valid_portfolio)
            print("Assets on portfolio: ", n)
            trading_data[t]["n"] = n
            trading_data[t]["names"] = []
            trading_data[t]["cov"] = np.zeros((n, n))

            trading_data[t]["er"] = np.zeros(n)
            trading_data[t]["er_weights"] = []
            trading_data[t]["er_sharpe_weights"] = []

            trading_data[t]["rr"] = np.zeros(n)

            tocov = np.zeros((n, estimate_size))
            for i, rs in enumerate(valid_portfolio):
                name = rs.get_stock_name()
                ts = rs.get_return_series()

                er, rr = wrapper_estimate_sw(rs, ts, offset=estimate_size+t)
                estimate_data[name]["er"] = np.append(estimate_data[name]["er"],
                                                      er)
                estimate_data[name]["rr"] = np.append(estimate_data[name]["rr"],
                                                      rr)

                # print()
                # print(name)
                # print("er: ", len(estimate_data[name]["er"]),
                #       "rr: ", len(estimate_data[name]["rr"]))
                er, rr, tocov[i, :] = wrapper_estimate_es(t, rs, estimate_data)
                trading_data[t]["names"].append(name)
                trading_data[t]["er"][i] = er
                trading_data[t]["rr"][i] = rr
            trading_data[t]["cov"] = np.cov(tocov)

            trading_data[t] = self._process(trading_data[t], risk_free)

            filename = "predict_pareto_" + str(t)
            pareto(trading_data[t], save=True, filename=filename)

            er = trading_data[t]["er"]
            rr = trading_data[t]["rr"]
            c = trading_data[t]["cov"]
            w = trading_data[t][w_type]

            trading_data[t]["p_r"] = np.dot(er, w)
            trading_data[t]["p_variance"] = np.dot(np.dot(w.T, c), w)
            trading_data[t]["p_std"] = np.sqrt(trading_data[t]["p_variance"])

            if t == 0:
                trading_data[t]["e_ar"] = calc_ar(er, w)
                trading_data[t]["r_ar"] = calc_ar(rr, w)

                trading_data[t]["pcm"] = 0.
                trading_data[t]["ti"] = 0.

            else:
                r_ar = trading_data[t-1]["r_ar"]
                pcm_1 = trading_data[t-1]["pcm"]

                trading_data[t]["e_ar"] = calc_ar(er, w, r_ar)
                trading_data[t]["r_ar"] = calc_ar(rr, w, r_ar)

                names = list(set(trading_data[t]["names"]) &
                             set(trading_data[t-1]["names"]))

                adj_w = np.zeros(len(names))
                adj_w_1 = np.zeros(len(names))
                adj_rr = np.zeros(len(names))
                for i, n in enumerate(names):
                    index = trading_data[t]["names"].index(n)
                    adj_w[i] = trading_data[t][w_type][index]
                    adj_rr[i] = rr[index]

                    index = trading_data[t-1]["names"].index(n)
                    adj_w_1[i] = trading_data[t-1][w_type][index]

                trading_data[t]["pcm"] = calc_pcm(adj_rr, adj_w, adj_w_1, pcm_1)
                trading_data[t]["ti"] = calc_ti(adj_w, adj_w_1)

            print(trading_data[t]["names"])
            print(w.T)
            print(t, "e_ar", trading_data[t]["e_ar"])
            print(t, "r_ar", trading_data[t]["r_ar"])
            print(t, "pcm", trading_data[t]["pcm"])
            print(t, "ti", trading_data[t]["ti"])

            # pprint.pprint(trading_data[t])
            # total_er.append(trading_data[t]["er"])
            # total_rr.append(trading_data[t]["rr"])
            #
            # error = mltools.Error(expected=total_er,
            #                       predicted=total_rr)
            # error.print_errors()

        plot_ar(trading_data)