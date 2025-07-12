import math
import numpy as np
import pandas as pd

from functions.assetClasses.options import ContractSpecs


def apply_interest(principal: float, rate: float, period: int, is_discrete: bool = True,
                   compounding_frequency: int = np.inf) -> float:
    """
    This function is used to apply the interest rate for a given principal, the result is the
    principal plus the interest

    :param principal:
    :param rate:
    :param period:
    :param is_discrete:
    :param compounding_frequency:
    :return:
    """

    def discrete(p: float, r: float, m: int, n: int) -> float:
        return p * math.pow(1 + r / m, m * n)

    def continuous(p: float, r: float, n: int) -> float:
        return p * math.exp(r * n)

    return (discrete(principal, rate, compounding_frequency, period) if is_discrete
            else continuous(principal, rate, period))


class RatesUtil:
    """
    This class is a utility class for rates conversion.
        1) Rates conversion between discrete compounding
        2) Rates conversion between discrete and continuous compounding

    For organization purpose only
    """
    @staticmethod
    def conversion_btw_discrete(rate: float, from_compounding_frequency: int, to_compounding_frequency: int,) -> float:
        """
        This function convert rate to equivalent from one compounding frequency to another.

        :param rate:
        :param from_compounding_frequency:
        :param to_compounding_frequency:
        :return:
        """
        _power: float = from_compounding_frequency / to_compounding_frequency
        return to_compounding_frequency * ((1 + rate / from_compounding_frequency)**_power -1)

    @staticmethod
    def conversion_btw_discrete_continuous(rate: float, compounding_frequency: int, compounding: bool = False) -> float:
        """
        This function convert rate to equivalent i.e., if compounding is True, the function returns
        discrete interest rate otherwise it's return the continuous interest rate.

        :param rate:
        :param compounding_frequency:
        :param compounding:
        :return:
        """
        if compounding:
            return compounding_frequency * (math.exp(rate / compounding_frequency) - 1)
        else:
            return compounding_frequency * math.log(1 + rate / compounding_frequency)

    @staticmethod
    def get_percentage_return(investment: float, payout: float, period: float=1, compounding: float=1,
                              is_continuous = False) -> float:

        """
        Given an initial investment and a payout, the function returns the percentage return.

        :param investment:
        :param payout:
        :param period:
        :param compounding:
        :param is_continuous:
        :return:
        """
        if is_continuous:
            return math.log(payout / investment) * (1/period)
        else:
            return compounding * ((payout / investment) ** (1/(period*compounding)) - 1)



class CorporateActions:

    @staticmethod
    def stock_split(strike_price: float, ratio: float, option_type: str = 'C') -> ContractSpecs:
        option_contract: ContractSpecs = ContractSpecs(strike_price, option_type=option_type)

        return ContractSpecs(option_contract.get_strike() * (ratio ** (-1)),
                             delivery=int(option_contract.get_delivery() * ratio))

    @staticmethod
    def stock_dividend(strike_price: float, stock_div: float, option_type: str = 'C') -> ContractSpecs:
        stock_split_ratio: float = 1 + stock_div
        return CorporateActions.stock_split(strike_price, stock_split_ratio, option_type)


def ttest_2_samples(sample1: tuple, sample2: tuple, mean_diff: float = 0.0) -> ():

    mu_1: float = sample1[0]
    mu_2: float = sample2[0]
    var_1: float = sample1[1] ** 2
    var_2: float = sample2[1] ** 2
    n_1: float = sample1[2]
    n_2: float = sample2[2]

    if len(sample1) != len(sample2):
        raise Exception('Number of argument must have equal length')
    elif sample1[2] != sample2[2]:
        t: float = ((mu_1 - mu_2) - mean_diff) / ((var_1 / n_1 + var_2 / n_2) ** 0.5)
        denominator: float = ((var_1 / n_1) ** 2 / n_1) + ((var_2 / n_2) ** 2 / n_2)
        df: float = ((var_1 / n_1 + var_2 / n_2) ** 2)/denominator
        return t, df
    else:

        pool_variance: float = (((n_1 - 1) * var_1 + (n_2 - 1) * var_2) / (n_1 + n_2 - 2))
        t: float = ((mu_1 - mu_2) - mean_diff) / ((pool_variance / n_1 + pool_variance / n_2) ** 0.5)
        return t, pool_variance, n_1 + n_2 - 2
