# -*- coding: utf-8 -*-

"""
    This file contains Portofolio class and developed methods.
"""

from .stock import Stock

from .regressors.regressor import Regressor
from .regressorstock import RegressorStock

import os
import numpy as np
from scipy import linalg
import datetime

try:
    from scipy import optimize
except ImportError:
    _SCIPY_AVAILABLE = 0
else:
    _SCIPY_AVAILABLE = 1

try:
    from scipy import stats
except ImportError:
    _SCIPY_STATS_AVAILABLE = 0
else:
    _SCIPY_STATS_AVAILABLE = 1

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


def init_error():
    print("Error: type of object must be a RegressorStock or a list of "
          "RegressorStock.")


def get_data(stock_name):
    """
    """    
    if not _PANDAS_AVAILABLE:
        raise Exception("Please install 'pandas' library to get data.")

    data_path = "data/"
    file_name = data_path + "aw_" + stock_name + ".csv"

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # If data dont exist, try to download and return it
    if not os.path.exists(file_name):
        try:
            start = datetime.datetime(2007, 1, 1)

            stock = web.DataReader(stock_name, "yahoo", start=start)

            # Get only wednesday data ( day 2 of the week )
            # If wednesday value is not available (eg. some holiday), search for
            # prior available value
            not_available = stock[(stock.index.dayofweek == 2) &
                                  (stock["Volume"] == 0)]
            new_days = []
            for day in not_available.index:
                new_day = day
                while stock["Volume"][new_day] == 0:
                    new_day = new_day - Day()
                new_days.append(new_day)

            tempa = stock.loc[new_days]
            tempb = stock[((stock.index.dayofweek == 2) & (stock["Volume"] != 0))]

            stock = tempa.append(tempb)
            st = stock.sort()

            st['Adj Close'].to_csv(file_name, index=False)

        except:
            raise Exception("Error: can't find ", stock_name)

    stock = Stock(name=stock_name, file_name=file_name)
    if stock.name == "r1" or stock.name == "s1":
        stock.returns = stock.time_series_data
        # stock.calculate_return()
    # print(stock.returns)
    
    return stock


class PortfolioBase:
    """
            Portfolio class
    """

    def __init__(self, regressorstock=None):

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

    # def __exist(self, stock_name):
    #     """ Find for stock in portfolio, if exists return a pointer to it,
    #         if dont returns False.
    #     """
    #
    #     for s in self.portfolio:
    #         if s.stock.name == stock_name:
    #             return s
    #     return False

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
        print("Risk - ", np.dot(np.dot(self.weights, self.cov),
                                   self.weights.T))

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

    def plot(self, stock_name):

        rs = self._exist(stock_name)
        ts = rs.get_time_series()
        r = rs.get_return_series()

        # acf = stats.tsa.stattools.acovf

        import matplotlib.pyplot as plt

        # plt.figure()
        f, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.plot(ts)
        ax1.set_title("Time Series")

        ax2.plot(r)
        ax2.set_title("Returns")

        ax3.acorr(r, usevlines=True, normed=True, maxlags=None, lw=2)
        ax3.grid(True)
        ax3.axhline(0, color='black', lw=2)
        interval = 1.96/np.sqrt(len(r))
        ax3.axhline(interval, color='red', lw=2)
        ax3.axhline(-interval, color='red', lw=2)
        ax3.set_xlim([-1, len(r)])
        ax3.set_title("Returns Autocorrelation")

        plt.show()
        plt.close()


    # def add(self, stock_name, regressor_name="elmk", window_size=4,
    #         dataprocess=None):
    #     """
    #         Add a stock to portofolio.
    #     """
    #
    #     if type(stock_name) is not str:
    #         raise Exception("stock_name must be a 'str' asset name from "
    #                         "BOVESPA.")
    #
    #     if self.__exist(stock_name):
    #         print("Warning: Couldn't add ", stock_name, " to portfolio ",
    #               "because stock already exists.")
    #         return
    #
    #     # Create a temporary stock object
    #     stock = get_data(stock_name)
    #     # stock.create_database(window_size=window_size, series_type="return")
    #
    #     # Create a temporary regressor object
    #     regressor = Regressor(regressor_type=regressor_name)
    #
    #     # Create a regressorstock object and add it to the portfolio list
    #     regressor = RegressorStock(regressor=regressor, stock=stock)
    #
    #     regressor.create_model_it("return", window_size)
    #
    #     self.portfolio.append(regressor)

    def remove(self, stock_name):
        """
            Remove a stock from portfolio.
        """

        if not self.__exist(stock_name):
            print("Warning: Couldn't remove ", stock_name, " from portfolio ",
                  "because stock don't exists.")
            return

        self.portfolio.remove(self.__exist(stock_name))
        
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
                # er = sr.predict(horizon=1)
                er = sr.predict_it(horizon=1)
                tri_std = sr.tri_result.get_std() ** 2
                tri_rmse = sr.tri_result.get("rmse")
                cv_rmse = sr.regressor.regressor.cv_best_error
                # print(sr.tri_result.get_anderson())
                # print(sr.tri_result.get_shapiro())

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
                # er = sr.predict(horizon=1)
                er = sr.predict_it(horizon=1)
                tri_std = sr.tri_result.get_std() ** 2
                tri_rmse = sr.tri_result.get("rmse")
                cv_rmse = sr.regressor.regressor.cv_best_error

                print(sr.stock.name, sr.regressor.type, er[0], tri_std,
                      cv_rmse, tri_rmse)
            print()

    # def __pareto(self, weights, assets):
    #     """
    #         Plot a pareto with each asset and all calculated portfolio
    #         possibilities.
    #     """
    #
    #     if not _MATPLOT_AVAILABLE:
    #         raise Exception("Please install 'matplotlib' to plot pareto.")
    #
    #     plt.figure(figsize=(24.0, 12.0))
    #
    #     # Plot assets data
    #     for a in range(len(assets["name"])):
    #         plt.plot(assets["cov"][a][a], assets["er"][a], "mo")
    #         plt.text(assets["cov"][a][a], assets["er"][a], assets["name"][a])
    #
    #     # Plot portfolio data
    #     for we in weights:
    #         p_return = np.dot(we.T, self.er)
    #         p_risk = np.dot(np.dot(we.T, self.cov), we)
    #
    #         plt.plot(p_risk, p_return, "bo")
    #
    #     p_return = np.dot(self.max_sharpe_weights, self.er)
    #     p_risk = np.dot(np.dot(self.max_sharpe_weights, self.cov),
    #                        self.max_sharpe_weights.T)
    #
    #     plt.plot(p_risk, p_return, "ro")
    #
    #     plt.title("Portfolio Pareto")
    #     plt.xlabel("Portfolio risk")
    #     plt.ylabel("Expected Portfolio return")
    #
    #     plt.show()
    #
    #     plt.close()

    # def _pareto(self, weights, assets):
    #     """
    #         Plot a pareto with each asset and all calculated portfolio
    #         possibilities.
    #     """
    #
    #     if not _MATPLOT_AVAILABLE:
    #         raise Exception("Please install 'matplotlib' to plot pareto.")
    #
    #     plt.figure(figsize=(24.0, 12.0))
    #
    #     # Plot assets data
    #     if len(self.portfolio) > 1:
    #         for a in range(len(assets["name"])):
    #             plt.plot(assets["cov"][a][a], assets["er"][a], "mo")
    #             plt.text(assets["cov"][a][a], assets["er"][a], assets["name"][a])
    #     else:
    #         plt.plot(assets["cov"], assets["er"][0], "mo")
    #         plt.text(assets["cov"], assets["er"][0], assets["name"][0])
    #
    #     print("assets", type(assets["er"]), assets["er"])
    #
    #     # Plot portfolio data
    #     for we in weights:
    #         p_return = np.dot(we.T, assets["er"])
    #         p_risk = np.dot(np.dot(we.T, assets["cov"]), we)
    #
    #         plt.plot(p_risk, p_return, "bo")
    #
    #     # if len(self.portfolio) > 1:
    #     #     p_return = np.dot(self.max_sharpe_weights, self.er)
    #     #     p_risk = np.dot(np.dot(self.max_sharpe_weights, self.cov),
    #     #                     self.max_sharpe_weights.T)
    #     # else:
    #     #     p_return = self.er[0]
    #     #     p_risk = self.cov
    #     #
    #     # plt.plot(p_risk, p_return, "ro")
    #
    #     plt.title("Portfolio Pareto")
    #     plt.xlabel("Portfolio risk")
    #     plt.ylabel("Expected Portfolio return")
    #
    #     plt.show()
    #
    #     plt.close()

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
        # Solve weights by maximizing sharpe ratio
        self.max_sharpe_weights = self.__solve_sharpe(self.er, self.cov, rf)

        # Solve quadratic problem defined in Fred 09 paper
        weights = []
        for erd in np.linspace(np.min(self.er), np.max(self.er), 50):
            weights.append(self.__solve_cvxopt(self.er, self.cov, erd))

        # Plot pareto
        assets = {"name": names, "er": self.er, "cov": self.cov}
        self.__pareto(weights, assets)

        self.__print_weights_sharpe(self.max_sharpe_weights)
        # self.__print_results()

    def calc_portfolio_metrics(self, data):
        # Accumulated return
        ar = 1

        # PCM
        pcm = 0

        # TI
        ti = 0

        for t in range(len(data)):
            ar *= 1 + np.dot(data[t]["er"], data[t]["weights"])

            for s in range(len(self.portfolio)):
                if t > 0:
                    pcm += np.dot(data[t]["er"][s], (data[t]["weights"][s] -
                                                     data[t-1]["weights"][s]))
                    if data[t]["weights"][s] != 0 and \
                        data[t-1]["weights"][s] != 0:
                        ti += np.fabs(data[t]["weights"][s] -
                                      data[t-1]["weights"][s])

        return ar, pcm, ti

    def test(self):

        trading_samples = 142
        estimating_samples = 103

        # cov_errors = self.__get_cov(risk="errors")

        # i = 0

        pr_error = np.empty((len(self.portfolio), estimating_samples))
        pr_er = np.empty(len(self.portfolio))

        cl_returns = np.empty((len(self.portfolio), estimating_samples))
        for t in range(trading_samples):

            for rs in range(len(self.portfolio)):
                jump = self.portfolio[rs].tri_result.predicted_targets.size -\
                       trading_samples - estimating_samples

                # print("[", jump + t, " : ", jump+t+estimating_samples-1, "]")

                pr_error[rs, :] = self.portfolio[rs].tri_result.get_error()\
                    [jump + t:jump + t + estimating_samples]

                pr_er[rs] = self.portfolio[rs].tri_result.predicted_targets\
                    [jump + t + estimating_samples]

                cl_returns[rs, :] = self.portfolio[rs].tri_result.\
                    expected_targets[jump + t:jump + t + estimating_samples]


            # Prediction model
            pr_cov = np.cov(pr_error)

    def _process(self, trading_data_frame, risk_free):
        """"""

        if len(trading_data_frame["names"]) > 1:
            trading_data_frame["er_weights"] = pareto_weights(trading_data_frame,
                                                              target="er")
            trading_data_frame["er_sharpe_weights"] = \
                solve_sharpe(trading_data_frame["er"],
                             trading_data_frame["cov"],
                             risk_free)

            min_risk = 999999.9
            max_return = -999999.9
            for w in trading_data_frame["er_weights"]:
                p_return = np.dot(w.T, trading_data_frame["er"])
                p_risk = np.dot(np.dot(w.T, trading_data_frame["cov"]), w)

                if p_risk < min_risk:
                    min_risk = p_risk
                    trading_data_frame["er_min_risk_weights"] = w
                    # print("min_risk: ", min_risk, trading_data_frame["er_min_risk_weights"])
                if p_return > max_return:
                    max_return = p_return
                    trading_data_frame["er_max_return_weights"] = w
                    # print("max_ret: ", max_return, trading_data_frame["er_max_return_weights"])

        # If there is only 1 asset on portfolio
        else:
            p_return = np.dot([1.], trading_data_frame["er"])

            if p_return > 0.:
                trading_data_frame["er_weights"] = np.ones((1, 1))
                trading_data_frame["er_sharpe_weights"] = np.ones((1, 1))
                trading_data_frame["er_min_risk_weights"] = np.ones((1, 1))
                trading_data_frame["er_max_return_weights"] = np.ones((1, 1))
            else:
                trading_data_frame["er_weights"] = np.zeros((1, 1))
                trading_data_frame["er_sharpe_weights"] = np.zeros((1, 1))
                trading_data_frame["er_min_risk_weights"] = np.zeros((1, 1))
                trading_data_frame["er_max_return_weights"] = np.zeros((1, 1))

# trading_data_frame["rr_weights"] = pareto_weights(trading_data_frame,
        #                                                   target="rr")
        # trading_data_frame["rr_sharpe_weights"] = \
        #     solve_sharpe(trading_data_frame["rr"],
        #                  trading_data_frame["cov"],
        #                  risk_free)
        #
        # min_risk = 999999.9
        # max_return = -999999.9
        # for w in trading_data_frame["rr_weights"]:
        #     p_return = np.dot(w.T, trading_data_frame["rr"])
        #     p_risk = np.dot(np.dot(w.T, trading_data_frame["cov"]), w)
        #
        #     if p_risk < min_risk:
        #         trading_data_frame["rr_min_risk_weights"] = w
        #
        #     if p_return > max_return:
        #         trading_data_frame["rr_max_return_weights"] = w

        return trading_data_frame


def solve_sharpe(r, cov, risk_free=0.00025):
    """
        Search through an optimization process the weights of each asset
        that maximizes the value of a Sharpe ratio.

    """

    if not _SCIPY_AVAILABLE:
        raise Exception("Please install 'scipy' library to optimize sharpe"
                        " ratio.")

    if type(r) is not np.ndarray:
        r = np.array(r)
    r = r.reshape(-1, 1)

    def f(x, r, cov, rf):
        # utility = Sharpe ratio
        util = (np.dot(x.T, r) - rf)/np.sqrt(np.dot(np.dot(x.T, cov), x))

        # maximize the utility, minimize its inverse value
        return 1 / util

    # Problem Constraints
    # Weights between 0 and 1
    boundaries = [(0., 1.) for i in range(r.size)]

    # Weight sum equals to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.})

    weights = []
    optimized = []
    sucess = False
    for it in range(100):
        if it == 0:
            x = np.ones(r.shape) / r.size

        # If first attempt failed, let's try a new initial weights guess
        else:
            x = np.random.uniform(0, 1, r.size).reshape(-1, 1)
            x = x/np.sum(x)

        optimized = optimize.minimize(f, x, (r, cov, risk_free),
                                      method='SLSQP', constraints=constraints,
                                      bounds=boundaries)

        weights = optimized.x
        # weights[np.isclose(weights, [0], atol=1e-6)] = 0
        if optimized.success and \
                np.isclose(np.sum(weights), [1]) and \
                not np.any(weights > 1) and \
                not np.any(weights < 0):
            sucess = True
            break

    if not optimized.success:
        print("Error 1: Could not find max sharpe weights. Returned 0's.")
        return np.zeros(r.shape)
        # raise BaseException(optimized.message)

    if not sucess:
        if np.any(weights < 0):
            weights[weights < 0] = 0

        if not np.isclose(np.sum(weights), [1]):
            weights = weights/np.sum(weights)

    x = weights
    sharpe = (np.dot(x.T, r) - risk_free)/np.sqrt(np.dot(np.dot(x.T, cov), x))
    if sharpe < 0:
        print("Error 2: Could not find a positive sharpe. Returned 0's.")
        return np.zeros(r.shape)

    # Set to zero weights minor than 1e-6
    weights[np.isclose(weights, [0], atol=1e-6)] = 0.
    weights = weights/np.sum(weights)

    return weights


def solve_sharpe_cvxpy(er, cov, risk_free=0.00025):
    """
        Find through an optimization the weights of each asset that
        maximizes the value of a Sharpe ratio.

    """

    r = np.array(er).reshape(-1, 1)

    c_sqrt = linalg.sqrtm(cov)

    import cvxpy as cvx

    # Construct the problem.

    y = cvx.Variable(len(er))
    obj = cvx.quad_form(y, cov)
    constraints = [(r - np.ones(r.shape)*risk_free).T * y == 1,
                   y > 0]
    prob = cvx.Problem(cvx.Minimize(obj), constraints)
    prob.solve()

    weights = y.value
    if weights is not None:
        weights = np.squeeze(np.asarray(weights))

        weights = weights/np.sum(weights)
        # weights[np.isclose(weights, [0], atol=1e-6)] = 0
        # weights = weights/np.sum(weights)
    else:
        print("Error 1: Could not find a positive sharpe. Returned 0's.")
        return np.zeros(r.shape)

    return weights


def solve_cvxopt(er, cov, er_d):
        """
            Find through an optimization of a quadratic problem the weights of
            each asset that minimizes the risk based upon a desired return.

        """

        if not _CVXOPT_AVAILABLE:
            raise Exception("Please install 'cvxopt' library to solve "
                            "optimization problem.")

        solvers.options['show_progress'] = False

        P = matrix(2 * cov, tc="d")
        q = matrix(np.zeros(len(er)), tc="d")
        G = matrix(-1 * np.eye(len(er)), tc="d")
        h = matrix(np.zeros(len(er)), tc="d")

        A = np.array(er).reshape((1, -1))
        A = np.concatenate((A, np.ones((1, len(er)))))

        A = matrix(A, tc="d")
        b = matrix([er_d, 1], tc="d")

        sol = solvers.qp(P, q, G, h, A, b)

        x = np.array(sol["x"])

        # Set to zero weights minor than 1e-6
        x[np.isclose(x, [0], atol=1e-6)] = 0
        x = x/np.sum(x)

        return x


def pareto(trading_data_frame, plot=False, save=True, filename=None):
    """
        Plot a pareto with each asset and all calculated portfolio
        possibilities.
    """

    if not _MATPLOT_AVAILABLE:
        raise Exception("Please install 'matplotlib' to plot pareto.")

    # print("er: ", trading_data["er"])
    # print("rr: ", trading_data["rr"])

    plt.figure(figsize=(24.0, 12.0))

    # Plot assets data
    if len(trading_data_frame["names"]) > 1:
        pass
        # for a in range(len(trading_data_frame["names"])):
        #     plt.plot(trading_data_frame["cov"][a][a],
        #              trading_data_frame["er"][a], "mo")
        #
        #     plt.text(trading_data_frame["cov"][a][a],
        #              trading_data_frame["er"][a],
        #              trading_data_frame["names"][a])

    else:
        print("Need at least 2 assets to plot pareto.")
        return

    # print("assets", type(trading_data["er"]), trading_data["er"])

    r = trading_data_frame["er"]
    c = trading_data_frame["cov"]

    # Plot portfolio data
    for w in trading_data_frame["er_weights"]:
        p_return = np.dot(w.T, r)
        p_risk = np.dot(np.dot(w.T, c), w)

        plt.plot(p_risk, p_return, "bo")
        plt.errorbar(p_risk, p_return, yerr=np.sqrt(p_risk), ecolor="b")

    w = trading_data_frame["er_max_return_weights"]
    p_return = np.dot(w.T, r)
    p_risk = np.dot(np.dot(w.T, c), w)
    plt.plot(p_risk, p_return, "ko", label="Max Return Portfolio")
    plt.legend()

    w = trading_data_frame["er_min_risk_weights"]
    p_return = np.dot(w.T, r)
    p_risk = np.dot(np.dot(w.T, c), w)
    plt.plot(p_risk, p_return, "go", label="Min Risk Portfolio")
    plt.legend()

    w = trading_data_frame["er_sharpe_weights"]
    p_return = np.dot(w.T, r)
    p_risk = np.dot(np.dot(w.T, c), w)
    plt.plot(p_risk, p_return, "ro", label="Max Sharpe Portfolio")
    plt.legend()

    plt.title("Portfolio Pareto")
    plt.xlabel("Portfolio risk")
    plt.ylabel("Expected Portfolio return")

    if plot:
        plt.show()

    if save:
        if filename is None:
            print("Set filename to save pareto plot.")
            return

        filename += ".png"
        plt.savefig(filename)

    plt.close()


def calc_ar(returns, weights, ar=None):

    result = 1 + np.dot(returns, weights)
    # result = result[0]

    if ar is not None:
        result *= ar

    # print()
    # print("#########")
    # print(type(result))

    return result


def calc_pcm(returns, weights_t, weights_t_1, pcm):

    result = np.dot(returns, (weights_t - weights_t_1))

    if pcm is not None:
        result += pcm

    return result


def calc_ti(weights_t, weights_t_1):

    mask = np.logical_and(weights_t != 0., weights_t_1 != 0.)

    return np.sum(np.fabs(weights_t[mask] - weights_t_1[mask]))


def calc_portfolio_metrics(trading_data, re="er", we="sharpe"):
    # Accumulated return
    ar = 1

    # PCM
    pcm = 0

    # TI
    ti_list = []

    if we == "sharpe":
        er_we = "er_sharpe_weights"

    elif we == "min_risk":
        er_we = "er_min_risk_weights"

    elif we == "max_return":
        er_we = "er_max_return_weights"

    else:
        print("Error: Invalid weight.")
        return

    p_return = 0
    for t in range(len(trading_data)):
        p_return = np.dot(trading_data[t][re], trading_data[t][er_we])
        # print(t, "p_return", p_return)
        # print(trading_data[t][re])
        # print(trading_data[t][er_we])
        ar *= 1 + p_return

        print(t, "ar ", ar)
        ti = 0
        for s in range(len(trading_data[t]["names"])):
            if t > 0:
                pcm += np.dot(trading_data[t][re][s],
                              (trading_data[t][er_we][s] -
                               trading_data[t-1][er_we][s]))
                # print(pcm, trading_data[t][er_we][s], trading_data[t-1][er_we][s])

                if trading_data[t][er_we][s] != 0 and \
                        trading_data[t-1][er_we][s] != 0:

                    ti += np.fabs(trading_data[t][er_we][s] -
                                  trading_data[t-1][er_we][s])
        ti_list.append(ti)

    # print(ti_list)
    ti = np.mean(ti_list)
    # print(ti)
    metrics = {"ar": ar, "pcm": pcm, "ti": ti}
    return metrics


def pareto_weights(trading_data_frame, target="er"):

    # Solve quadratic problem defined in Fred 09 paper

    weights = []
    for erd in np.linspace(0,
    # for erd in np.linspace(np.min(trading_data_frame[target]),
                           np.max(trading_data_frame[target]), 30):

        weights.append(solve_cvxopt(trading_data_frame[target],
                                    trading_data_frame["cov"], erd))

    return weights


def gof_estimate_data(data):
    """"""

    if not _SCIPY_STATS_AVAILABLE:
            raise ImportError("Need 'scipy' module to calculate "
                              "anderson-darling  and shapiro-wilk test.")

    # mean = np.mean(data)
    # std = np.std(data)
    #
    # m = 3
    # outliers = np.logical_or((data < mean - m * std), (data > mean + m * std))
    # print(mean)
    # print(std)
    # print(outliers)
    #
    # data = data[np.logical_not(outliers)]
    #
    # print(data)

    # d = np.abs(data - np.median(data))
    # mdev = np.median(d)
    # s = d/mdev if mdev else 0.
    # data = data[s < 5.]
    # print(d)
    # print(mdev)
    # print(s)
    # print(data)

     # Calculate Anderson-Darling normality test index
    ad_statistic, ad_c, ad_s = stats.anderson(data, "norm")
    if ad_statistic > ad_c[4]:
        ad_result = False
    else:
        ad_result = True

    # print(ad_statistic, ad_c[4], ad_result)

    # Calculate Shapiro-Wilk normality index
    sw_statistic, sw_p_value = stats.shapiro(data)
    if sw_p_value > .01:
        sw_result = True
    else:
        sw_result = False

    # print(sw_statistic, sw_p_value, sw_result)

    result = {"ad_statistic": ad_statistic, "ad_result": ad_result,
              "sw_statistic": sw_statistic, "sw_result": sw_result,
              "result": ad_result and sw_result}

    # if not result["result"]:
    #     import matplotlib.pyplot as plt
    #     a, b = stats.probplot(data, plot=plt)
    #     # print(a)
    #     # print(b)
    #     plt.show()

    return result


def plot_ar(trading_data):

    if not _MATPLOT_AVAILABLE:
        raise Exception("Please install 'matplotlib' to plot pareto.")

    p_std = [3*trading_data[i]["p_std"][0][0]
             for i in list(trading_data.keys())]
    e_ar = [trading_data[i]["e_ar"][0] for i in list(trading_data.keys())]
    r_ar = [trading_data[i]["r_ar"][0] for i in list(trading_data.keys())]

    plt.figure(figsize=(24.0, 12.0))

    x = np.arange(1, len(e_ar)+1).tolist()

    plt.plot(x, e_ar, "b", label="Expected Return")
    plt.errorbar(x, e_ar, yerr=p_std)
    plt.plot(x, r_ar, "r", label="Realized Return")
    plt.legend()

    plt.show()
    plt.close()


def is_normal(portfolio):

    for i, rs in enumerate(portfolio):
     n = len(portfolio)