#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    test_portfolio
    --------

"""


import pai


def test_portfolio():

    elmk = pai.Regressor("elmk")
    stock1 = pai.Stock("random1")

    elmr = pai.Regressor("elmr")
    stock2 = pai.Stock("random2")

    rs1 = pai.RegressorStock(elmk, stock1)
    rs2 = pai.RegressorStock(elmr, stock2)

    p = pai.Portfolio()
    p.add_rs(rs1)
    p.add_rs(rs2)

    p.list()