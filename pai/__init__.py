# -*- coding: utf-8 -*-

__author__ = 'Augusto Almeida'
__email__ = 'acba@cin.ufpe.br'
__version__ = '0.1.0'

from .stock import Stock
from .regressors import Regressor
from .regressorstock import RegressorStock

from .portfolio import PortfolioBase
# from . import regressors
from elm import mltools
from .classicportfolio import Classic
from .predictportfolio import Predict