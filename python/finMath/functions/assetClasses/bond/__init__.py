import math
import numpy as np
import pandas as pd


class BondPricing:
    """
    This class hold functions related to bond pricing
    """

    @staticmethod
    def bond_pricing_vanilla(par_value: float, year_to_maturity: float, coupon_rate: float, yield_rate: float,
                             payment_frequency: int = 2) -> float:
        """
        This is the regular vanilla bond pricing with a coupon rate and yield.

        :param par_value: this is the face value of the bond and usually $1000.00
        :param year_to_maturity:     a year to maturity of the bond
        :param coupon_rate: The coupon rate in decimal
        :param yield_rate:  The yield rate in decimal
        :param payment_frequency:   an integer value for payment frequency, usually 2 for semiannual
        :return:
        """

        # Discrete discounting factor method
        def factor( i : int ) -> float:
            return (1 / (1 + yield_rate / payment_frequency))**i

        num: float = payment_frequency * year_to_maturity
        pv_par_value: float =  par_value * factor(int(num))
        coupon_value: float = (coupon_rate * par_value)/payment_frequency

        # one can use the function provided by calculus to calculate the geometric series, the calculation is accurate
        # to dollar and cents
        if year_to_maturity <= 40:
            local_num: float = num
            value: float = 0.0
            while local_num > 0:
                value += factor(local_num)
                local_num -= 1
            value =coupon_value * value
            return value + pv_par_value
        else:
            return coupon_value * (1 - (yield_rate / payment_frequency) ** num)/(1 - yield_rate / payment_frequency) + pv_par_value

    @staticmethod
    def bond_pricing(zeros_df: pd.DataFrame, coupon: float, par_value: float, bond_yield: float = None,
                     first_maturity: float = None, n: int = None, difference: float = None,
                     seq_func=lambda a, n, d: a + (n - 1) * d) -> float:
        if zeros_df.shape[1] != 2:
            return np.inf * -1

        def pricing_from_zeros(row: pd.Series, c: float = coupon) -> float:
            return c * math.exp(-1 * row[0] * (row[1] / 100))

        if bond_yield is None:
            values_series = zeros_df.apply(pricing_from_zeros, axis=1)
            expiry_time: float = zeros_df.iat[zeros_df.shape[0] - 1, 0]
            expiry_rate: float = zeros_df.iat[zeros_df.shape[0] - 1, 1]

            return values_series.sum() + (par_value * math.exp(-1 * expiry_time * (expiry_rate / 100)))
        else:

            price: float = 0.0
            temp_n: int = n
            while n > 0:
                price += coupon * math.exp(-1 * seq_func(first_maturity, n, difference) * (bond_yield / 100))
                n -= 1
            return price + par_value * math.exp(-1 * seq_func(first_maturity, temp_n, difference) * (bond_yield / 100))

    @staticmethod
    def h_func(bond_yield: float, coupon: float, par_value: float, first_maturity: float, n: int,
               difference: float, bond_price: float = None, seq_func=lambda a, n, d: a + (n - 1) * d) -> float:

        temp_n: int = n
        value: float = 0.0

        while n > 0:
            value += math.exp(-1 * seq_func(first_maturity, n, difference) * (bond_yield / 100))
            n -= 1
        value *= coupon
        value += par_value * math.exp(-1 * seq_func(first_maturity, temp_n, difference) * (bond_yield / 100))
        return value - bond_price

    @staticmethod
    def h_func_prime(bond_yield: float, coupon: float, par_value: float, first_maturity: float,
                     n: int, difference: float,
                     bond_price: float = None, seq_func=lambda a, n, d: a + (n - 1) * d) -> float:

        temp_n: int = n
        value: float = 0.0

        while n > 0:
            a: float = seq_func(first_maturity, n, difference)
            value += -1 * a * math.exp(-1 * a * (bond_yield / 100))
            n -= 1
        value *= coupon
        a: float = seq_func(first_maturity, temp_n, difference)
        value += -1 * a * par_value * math.exp(-1 * a * (bond_yield / 100))
        return value

    @staticmethod
    def h_func_2nd_prime(bond_yield: float, coupon: float, par_value: float, first_maturity: float,
                         n: int, difference: float,
                         bond_price: float = None, seq_func=lambda a, n, d: a + (n - 1) * d) -> float:

        temp_n: int = n
        value: float = 0.0

        while n > 0:
            a: float = seq_func(first_maturity, n, difference)
            value += -1 * a ** 2 * math.exp(-1 * a * (bond_yield / 100))
            n -= 1
        value *= coupon
        a: float = seq_func(first_maturity, temp_n, difference)
        value += -1 * a ** 2 * par_value * math.exp(-1 * a * (bond_yield / 100))
        return value

    @staticmethod
    def zeros_by_bootstrap_method(bonds_details: pd.DataFrame, coupon_frequency: int = 2,
                                  time_delta: float = 0.5) -> pd.DataFrame:

        def f(price: float, coupon: float, par_value: float, time_to_maturity: float, coupon_freq: int) -> float:
            return math.log(price / (par_value + (coupon / coupon_freq))) / (-1 * time_to_maturity)

        def h(zeros_param_df: pd.DataFrame, i: int):

            value: float = 0.0
            time_to_maturity: float = bonds_details.at[i, 'Maturity']
            a_n: float = time_to_maturity
            coupon: float = bonds_details.at[i, 'Annual_coupon'] / coupon_frequency
            price: float = bonds_details.at[i, 'Bond_price']
            par_value: float = bonds_details.at[i, 'principal']

            time_to_maturity -= time_delta
            while time_to_maturity > 0.00:
                zero_yield: float = zeros_param_df[zeros_param_df.Maturity == time_to_maturity].to_numpy()[0][1]
                value += math.exp(-1 * zero_yield * time_to_maturity)
                time_to_maturity -= time_delta

            value *= coupon
            value = math.log((price - value) / (par_value + coupon)) / (-1 * a_n)
            return value

        zeros_df: pd.DataFrame = pd.DataFrame(columns=['Maturity', 'Zero_Rate'])
        num_of_rows: int = bonds_details.shape[0]
        index: int = 0

        while index < num_of_rows:

            if bonds_details.loc[index].iat[2] == 0:
                bond_yield: float = f(bonds_details.loc[index].iat[3], bonds_details.loc[index].iat[2],
                                      bonds_details.loc[index].iat[0], bonds_details.loc[index].iat[1],
                                      coupon_frequency)
                zeros_df.loc[len(zeros_df.index)] = [bonds_details.loc[index].iat[1], bond_yield]
            else:
                zeros_df.loc[len(zeros_df.index)] = [bonds_details.loc[index].iat[1], h(zeros_df, index)]

            index += 1

        return zeros_df

    @staticmethod
    def forward_rates(zero_rates_df: pd.DataFrame, time_delta: float) -> pd.DataFrame:

        def forward_rate_func(r2: float, r1: float, t2: float, t1: float) -> float:
            return r2 + (r2 - r1) * (t1 / (t2 - t1))

        forward_rates_df: pd.DataFrame = pd.DataFrame(columns=['Maturity', 'Zero Rates', 'Forward Rate'])
        num_of_rows: int = zero_rates_df.shape[0] - 1
        index: int = 0
        forward_rates_df.loc[len(forward_rates_df.index)] = [zero_rates_df.loc[index].iat[0],
                                                             zero_rates_df.loc[index].iat[1], 0.00]

        while index < num_of_rows:
            r_2: float = zero_rates_df.at[index + 1, 'Zero_Rate']
            r_1: float = zero_rates_df.at[index, 'Zero_Rate']
            t_2: float = zero_rates_df.at[index + 1, 'Maturity']
            t_1: float = zero_rates_df.at[index, 'Maturity']

            f_rate: float = forward_rate_func(r_2, r_1, t_2, t_1)
            forward_rates_df.loc[len(forward_rates_df.index)] = [zero_rates_df.loc[index + 1].iat[0],
                                                                 zero_rates_df.loc[index + 1].iat[1], f_rate]

            index += 1
        return forward_rates_df