# -*- coding: utf-8 -*-

"""
    This file contains Portofolio class and developed methods.
"""

from .stock import Stock
from .regressors.regressor import Regressor
from .regressorstock import RegressorStock

import os
import numpy as np
import datetime
from .portfolio import PortfolioBase
from .portfolio import get_data
from .portfolio import solve_sharpe
from .portfolio import solve_cvxopt
from .portfolio import solve_sharpe_cvxpy
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


class Classic(PortfolioBase):
    """
            Portfolio class
    """

    def __init__(self):
        super(self.__class__, self).__init__()

        self.portfolio = []

        # Weight from each asset on the portfolio
        self.weights = []

        # Weights from max Sharpe ratio point
        self.max_sharpe_weights = []
        self.max_sharpe = 0

        # Expected returns from each asset on the portfolio
        self.er = []

        # Covariance matrix from all assets on the portfolio
        self.cov = []

    def _exist(self, stock_name):
        """
            Search for stock in portfolio, if exists return it,
            if dont returns False.
        """

        for s in self.portfolio:
            if s.name == stock_name:
                return s
        return False

    def __get_cov(self):
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
        for s in self.portfolio:
            if s.returns.size < ret_size:
                ret_size = s.returns.size

        ret_matrix = np.empty((len(self.portfolio), ret_size))

        for s in range(len(self.portfolio)):
            ret_matrix[s, :] = self.portfolio[s].returns[-ret_size:]

        cov = np.cov(ret_matrix)
        # a = np.linalg.cholesky(cov)
        # print(a)

        return cov

    def add(self, stock_name):
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

        self.portfolio.append(stock)

    # def remove(self, stock_name):
    #     """
    #         Remove a stock from portfolio.
    #     """
    #
    #     if not self._exist(stock_name):
    #         print("Warning: Couldn't remove ", stock_name, " from portfolio ",
    #               "because stock don't exists.")
    #         return
    #
    #     self.portfolio.remove(self._exist(stock_name))
        
    def list(self):
        """
            Print a table with all assets.
        """

        if _PRETTYTABLE_AVAILABLE:

            table = PrettyTable(["Stock name",
                                 "Expected return",
                                 "Risk (return variance)"])
            table.align["Stock name"] = "l"

            print("Listing portfolio: \n")
            for s in self.portfolio:
                er = s.expected_return_mean(samples=0)
                risk = s.risk_std(samples=0) ** 2

                table.add_row([s.name, er, risk])

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

            for s in self.portfolio:
                er = s.expected_return_mean(samples=0)
                risk = s.risk_std(samples=0) ** 2

                print(s.name, er, risk)
            print()

    def __print_weights_sharpe(self, we):

        if _PRETTYTABLE_AVAILABLE:
            table = PrettyTable(["Stock name", "%"])
            table.align["Stock name"] = "l"

            print()
            for i, s in enumerate(self.portfolio):
                table.add_row([s.name,
                               np.around(100 * (we[i]/np.sum(we)), decimals=2)])

            print(table.get_string(sortby="Stock name"))
            print()

        else:
            print("Max Sharpe portfolio")
            print()
            print("Stock name | ", "%")
            for i, s in enumerate(self.portfolio):
                print(s.name, np.around(100 * (we[i]/np.sum(we)), decimals=2))

            print()

    def optimize(self):
        """
            Optimize the weights from current portfolio, find the max Sharpe
            ratio weight, a pareto of portfolios and plot it.
        """

        names = []
        self.er = []

        # Calculate expected return
        for s in self.portfolio:
            names.append(s.name)
            self.er.append(s.expected_return_mean(samples=0))

        # Calculate covariance (risk) between assets from portfolio
        self.cov = self.__get_cov()

        # Weekly risk-free rate (%)
        rf = 0.1193 / 100

        weights = []
        if len(self.portfolio) > 1:
            # Solve weights by maximizing sharpe ratio
            self.max_sharpe_weights = solve_sharpe(self.er, self.cov, rf)

            # Solve quadratic problem defined in Fred 09 paper
            for er_d in np.linspace(np.min(self.er), np.max(self.er), 50):
                weights.append(solve_cvxopt(self.er, self.cov, er_d))
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
        # for s in range(len(self.portfolio)):
        #     self.portfolio[s].returns = np.arange(1, 34)

        returns = np.empty((len(self.portfolio), estimating_samples))

        per = refper =1
        data = []
        ref_data = []
        for t in range(trading_samples):

            er = []
            ref_er = []
            for s in range(len(self.portfolio)):
                jump = self.portfolio[s].returns.size - \
                       trading_samples - estimating_samples

                if jump < 0:
                    print("Error: Not enough data points, "
                          " set a lower  'trading_samples' or "
                          "'estimating_samples'.")
                    raise Exception("Error: Not enough data.")

                # tmp = self.portfolio[s].returns\
                #     [jump + t:jump + t + estimating_samples]
                # print(jump, "[", jump + t, " : ", jump+t+estimating_samples-1,
                #       "]", tmp)

                returns[s, :] = self.portfolio[s].returns \
                    [jump + t:jump + t + estimating_samples]

                er.append(np.mean(returns[s, :]))
                ref_er.append(self.portfolio[s]. \
                    returns[jump + t + estimating_samples])
            cov = np.cov(returns)
            # try:
            #     np.linalg.cholesky(cov)
            # except np.linalg.LinAlgError:
            #     print("not positive definite")

            print("### START ####")
            print()
            # print("predicted")
            # sharpe_weights = solve_sharpe(er, cov, risk_free)
            # sh = (np.dot(sharpe_weights.T, er) - risk_free)/\
            #      np.sqrt(np.dot(np.dot(sharpe_weights.T, cov), sharpe_weights))
            # print("sharpe: ", sh)
            # print("sharpe_weights: ", sharpe_weights)
            # print("weights sum: ", np.sum(sharpe_weights))
            # print()

            sharpe_weights = solve_sharpe_cvxpy(er, cov, risk_free)

            # sh = (np.dot(sharpe_weights.T, er) - risk_free)/\
            #      np.sqrt(np.dot(np.dot(sharpe_weights.T, cov), sharpe_weights))
            # print("cvxpy sharpe: ", sh)
            # print("cvxpy sharpe_weights: ", sharpe_weights)
            # print("cvxpy weights sum: ", np.sum(sharpe_weights))
            # print()
            # if t == 30:
            #     exit()
                # print()
                # exit()
                # print("reference")
            # ref_sharpe_weights, sucess = solve_sharpe(ref_er, cov, risk_free)
            # if not sucess:
            #     ref_sharpe_weights = solve_sharpe_cvxpy(ref_er, cov, risk_free)

            # sh = (np.dot(ref_sharpe_weights.T, ref_er) - risk_free)/\
            #      np.sqrt(np.dot(np.dot(ref_sharpe_weights.T, cov), ref_sharpe_weights))
            # print("ref_sharpe: ", sh)
            # print("ref_sharpe_weights: ", ref_sharpe_weights)
            # print("ref_sharpe_weights sum: ", np.sum(ref_sharpe_weights))
            # print()

            ref_sharpe_weights = solve_sharpe_cvxpy(ref_er, cov, risk_free)

            # sh = (np.dot(ref_sharpe_weights.T, ref_er) - risk_free)/\
            #      np.sqrt(np.dot(np.dot(ref_sharpe_weights.T, cov), ref_sharpe_weights))
            # print("cvxpy ref_sharpe: ", sh)
            # print("cvxpy ref_sharpe_weights: ", ref_sharpe_weights)
            # print("cvxpy ref_sharpe_weights sum: ", np.sum(ref_sharpe_weights))
            # print()
            # print()

            # print()
            # per *= 1 + np.dot(er, sharpe_weights)
            # refper *= 1 + np.dot(ref_er, ref_sharpe_weights)
            # print("p_er: ", np.dot(er, sharpe_weights))
            # print("p_er_ref: ", np.dot(ref_er, ref_sharpe_weights))
            # print()
            # print("per: ", per)
            # print("refper: ", refper)
            # print()
            # print("ref: ", ref_er)
            # print("pred: ", er)

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

    def _estimate(self, rs, ts, z, p, dataprocess, search):

        # print()
        # print("_estimate")
        # print("mean from")
        print(ts[:-1])

        return np.mean(ts[:-1])

    def _get_tocov(self, estimate_er, estimate_rr):

        tocov = np.array(estimate_rr)

        return tocov

    def _get_er(self, estimate_er, estimate_rr, start, end):

        # print()
        # print("_get_er")
        # print("mean from")
        print(estimate_rr[start:end])
        er = np.mean(estimate_rr[start:end])

        return er

    def benchmark(self, sw, p, k, estimate_size, trading_size, dataprocess=None,
                  search=False, risk_free=0.1193 / 100, weights="sharpe"):

        # self.portfolio[0].returns = np.arange(1, 51)
        # self.portfolio[1].returns = np.arange

        def get_valid_portfolio(t):

            valid_portfolio = []
            valid_ad = []
            valid_sw = []

            for rs in self.portfolio:
                name = rs.get_stock_name()
                ts = rs.get_return_series()

                if len(ts) < trading_size + estimate_size + (sw-1) + k:
                    print(len(ts), trading_size + estimate_size + (sw-1) + k)
                    print("Error: Change parameters, not enough data to trade.")
                    return

                start = len(ts) - trading_size - estimate_size + t
                end = start + estimate_size

                estimate_set = ts[start:end]

                result = gof_estimate_data(estimate_set)
                if result["result"]:
                    valid_portfolio.append(rs)
                    valid_ad.append(result["ad_statistic"])
                    valid_sw.append(result["sw_statistic"])
                else:
                    print(t, "Removed ", name)

            mean_ad = np.mean(valid_ad)
            mean_sw = np.mean(valid_sw)

            gof = {"mean_ad": mean_ad, "mean_sw": mean_sw}

            return valid_portfolio, gof

        if weights == "max_return":
            w_type = "er_max_return_weights"

        elif weights == "min_risk":
            w_type = "er_min_risk_weights"

        elif weights == "sharpe":
            w_type = "er_sharpe_weights"

        else:
            print("Invalid weight type.")
            return

        import pprint

        print("")
        print("Trading")
        print("")

        trading_data = {}
        for t in range(trading_size):
            trading_data[t] = {}

            valid_portfolio, gof = get_valid_portfolio(t)

            n = len(valid_portfolio)
            print("Assets on portfolio: ", n)
            trading_data[t]["n"] = n

            trading_data[t]["ad"] = gof["mean_ad"]
            trading_data[t]["sw"] = gof["mean_sw"]

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

                start = len(ts) - trading_size - estimate_size + t
                end = start + estimate_size

                estimate_set = ts[start:end]

                er = np.mean(estimate_set)
                rr = ts[end]
                tocov[i, :] = estimate_set

                trading_data[t]["names"].append(name)
                trading_data[t]["er"][i] = er
                trading_data[t]["rr"][i] = rr
            trading_data[t]["cov"] = np.cov(tocov)

            trading_data[t] = self._process(trading_data[t], risk_free)

            filename = "classic_pareto_" + str(t)
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
            print(t, "p_std", trading_data[t]["p_std"])
            print(t, "e_ar", trading_data[t]["e_ar"])
            print(t, "r_ar", trading_data[t]["r_ar"])
            print(t, "pcm", trading_data[t]["pcm"])
            print(t, "ti", trading_data[t]["ti"])

            # pprint.pprint(trading_data[t])
            # error = mltools.Error(expected=total_er,
            #                       predicted=total_rr)
            #
            # error.print_errors()

            # print(estimate_data)

        plot_ar(trading_data)