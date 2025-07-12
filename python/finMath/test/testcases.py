import unittest
from scipy.stats import norm
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import RootResults
from functions import RatesUtil, apply_interest, CorporateActions, ttest_2_samples
from functions.assetClasses.bond import BondPricing
from functions.assetClasses.options import ContractSpecs, OptionContract, OptionsCalculator, option_payoff, SecEnum


class FunctionTest(unittest.TestCase):

    def test_equivalent_rate(self):
        m: int = 2
        interest_rate: float = 0.1

        print(RatesUtil.conversion_btw_discrete_continuous(0.0687, 2))
        print(RatesUtil.conversion_btw_discrete_continuous(0.20, 2))
        print(RatesUtil.conversion_btw_discrete_continuous(0.19, 12))

        self.assertEqual(RatesUtil.conversion_btw_discrete_continuous(interest_rate, m), 0.09758032833886408)
        self.assertEqual(RatesUtil.conversion_btw_discrete_continuous(0.09758032833886408, m, compounding=True), 0.10000000000000009)

    def test_apply_interest_rate(self):
        period: int = 1
        rate: float = 0.1
        principal: float = 100.0

        self.assertEqual(110.51709180756477, apply_interest(principal, rate, period, False))

    def test_bond_pricing(self):
        df = pd.DataFrame({'maturity': [0.5, 1.0, 1.5, 2.0], 'zero_rate': [6.76, 6.76, 6.76, 6.76]})
        p: float = BondPricing.bond_pricing(df, 3.0, 100, bond_yield=6.76,
                                            first_maturity=0.5, n=4, difference=0.5)
        print(df.shape[0])
        # print(BondPricing.h_func(6.76, 3, 100, 0.5, 4, 0.5, p))
        # print(BondPricing.h_func_prime(6.76, 3, 100, 0.5, 4, 0.5, p))

        val = opt.newton(func=BondPricing.h_func, x0=3.5, fprime=BondPricing.h_func_prime,
                         args=(3, 100, 0.5, 4, 0.5, p), maxiter=5000)
        print(val)

        val = opt.newton(func=BondPricing.h_func, x0=3.5, fprime=BondPricing.h_func_prime,
                         args=(3, 100, 0.5, 4, 0.5, p), maxiter=1550, fprime2=BondPricing.h_func_2nd_prime)
        print(val)

    def test_zeros_bootstrapping(self):
        bonds_df: pd.DataFrame = pd.DataFrame({'principal': [100.0, 100.0, 100.0, 100.0, 100.0],
                                               'Maturity': [0.25, 0.50, 1.00, 1.50, 2.00],
                                               'Annual_coupon': [0, 0, 0, 8, 12],
                                               'Bond_price': [97.5, 94.9, 90.0, 96., 101.6]})
        zeros_df: pd.DataFrame = BondPricing.zeros_by_bootstrap_method(bonds_df)
        print(zeros_df)

        zeros_test_df: pd.DataFrame = pd.DataFrame({'Maturity': [1, 2, 3, 4, 5],
                                                    'Zero_Rate': [3.0, 4.0, 4.6, 5.0, 5.3]})
        print(BondPricing.forward_rates(zeros_test_df))

    def test_corporate_actions(self):
        option_contract: ContractSpecs = CorporateActions.stock_split(100.00, 3 / 2)
        print(option_contract.get_delivery())
        print(option_contract.get_strike())

        option_contract: ContractSpecs = CorporateActions.stock_dividend(15.00, 0.25)
        print(option_contract.get_delivery())
        print(option_contract.get_strike())

    def test_naked_option_initial_margin(self):
        option_price: float = 3.50
        strike_price: float = 60.0
        underlying_price: float = 57.0
        option_type: str = 'C'
        option_specs: ContractSpecs = ContractSpecs(strike_price, option_type=option_type)
        option_contract: OptionContract = OptionContract(underlying_price, option_price, option_specs)

        print(OptionsCalculator.naked_option_initial_margin(option_contract, 5))

    def test_portfolio_construction(self):
        def func(additional_quantity_1: float, lambda_1: float, current_value: float, additional_amount: float,
                 price_1: float, quantity_1: float) -> float:
            return (lambda_1 * (current_value + additional_amount)) / price_1 - quantity_1 - additional_quantity_1

        def func_prime(additional_quantity_1: float, lambda_1: float, current_value: float, additional_amount: float,
                       price_1: float, quantity_1: float) -> float:
            return lambda_1 - 1

        def func_prime_2(additional_quantity_1: float, lambda_1: float, current_value: float, additional_amount: float,
                         price_1: float, quantity_1: float) -> float:
            return 0

        lambda_val: float = 0.65
        port_current_value: float = 116.0
        port_additional_amount: float = 100.0
        quantity_asset_1: float = 13.0
        price_asset_1: float = 7.0
        additional_quantity_asset_1: float = 112.0

        """val = opt.newton(func=func, x0=additional_quantity_asset_1, fprime=func_prime,
                         args=(lambda_val, port_current_value, port_additional_amount, price_asset_1, quantity_asset_1),
                         maxiter=1550, fprime2=func_prime_2)
        print(val)"""

        val = opt.bisect(f=func, a=0, b=additional_quantity_asset_1,
                         args=(lambda_val, port_current_value, port_additional_amount, price_asset_1, quantity_asset_1),
                         maxiter=1550)
        print(val)

    def test_portfolio_construction_vals(self):
        def func(additional_amt_asset_1: float, asset_1_val: float, current_value: float, additional_amount: float,
                 lambda_1: float) -> float:
            return (lambda_1 * (current_value + additional_amount)) - asset_1_val - additional_amt_asset_1

        def func_prime(additional_amt_asset_1: float, asset_1_val: float, current_value: float,
                       additional_amount: float, lambda_1: float) -> float:
            return -1 * 0.35

        lambda_val: float = 0.65
        port_current_value: float = 100.0
        port_additional_amount: float = 50.0
        quantity_asset_1: float = 15.0
        asset_1_value: float = 75.0
        add_amt_asset_1: float = 2.0

        val = opt.newton(func=func, x0=add_amt_asset_1, fprime=func_prime,
                         args=(asset_1_value, port_current_value, port_additional_amount, lambda_val),
                         maxiter=5550)
        print(val)

    def test_portfolio_construction_question(self):
        def func(additional_quantity_1: float, lambda_1: float, current_value: float, additional_amount: float,
                 price_1: float, quantity_1: float) -> float:
            return (lambda_1 * (current_value + additional_amount)) / price_1 - quantity_1 - additional_quantity_1

        lambda_val: float = 0.65
        port_current_value: float = 132.25
        port_additional_amount: float = 167.75
        quantity_asset_1: float = 13.0
        price_asset_1: float = 7.50
        additional_quantity_asset_1: float = 112.0

        val = opt.bisect(f=func, a=0, b=additional_quantity_asset_1,
                         args=(lambda_val, port_current_value, port_additional_amount, price_asset_1, quantity_asset_1),
                         maxiter=100)
        print(val)

    def test_naked_options(self):
        contract_spec: ContractSpecs = ContractSpecs(60.00, 'C')
        option_contract: OptionContract = OptionContract(57.00, 3.50, contract_spec)
        margin: float = OptionsCalculator.naked_option_initial_margin(option_contract, 5)
        print(margin)

    def test_option_payoff(self):
        strike: float = 52.00
        print(option_payoff(strike, 60, 'P'))
        print(option_payoff(strike, 70, 'P'))
        print(option_payoff(strike, 40, 'P'))

        print(option_payoff(21, 24.2, 'C'))
        print(option_payoff(21, 22, 'C'))

    def test_risk_neutral_probability(self):
        print(OptionsCalculator.risk_neutral_probability('STK', 2, 2, 0.3,
                                                         0.05))

        print(OptionsCalculator.risk_neutral_probability('INDX', 2, 0.5, 0.2,
                                                         0.05, 0.02))

        print('CURR', OptionsCalculator.risk_neutral_probability('CURR', 12, 1, 0.12,
                                                                 0.05, 0.08))
        print(OptionsCalculator.risk_neutral_probability('FUT', 3, 9 / 12, 0.30))

        value: dict = OptionsCalculator.options_multiplier(100, 110, 90, 0.08,
                                                           1, 2)
        print(value)
        print(value['p'])
        print(value['p'] ** 2)
        print(2 * value['p'] * (1 - value['p']))
        print((1 - value['p']) ** 2)

        print(np.exp(-2 * 0.08 * 0.5) * (
                value['p'] ** 2 * 0 + 2 * value['p'] * (1 - value['p']) * 1 + (1 - value['p']) ** 2 * 19))

    def test_sample(self):
        n_up: int = 1500
        n_down: int = 80
        mult: int = 1
        n_up2: int = n_up

        while n_up > n_down:
            mult *= n_up
            n_up -= 1

        print(mult - n_up2 ** n_down)

    def test_bs(self):
        print(norm.cdf(-0.7693))

        def d1(s: float, k: float, r: float, sigma: float, t: float) -> float:
            return (np.log(s / k) + (r + (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t))

        def d2(d1_val: float, sigma: float, t: float) -> float:
            return d1_val - sigma * np.sqrt(t)

        d1_value: float = d1(42, 40, 0.1, 0.2, 0.5)
        d2_value: float = d2(d1_value, 0.2, 0.5)
        k_dis: float = 40 * np.exp(-1 * 0.1 * 0.5)
        print(d1_value, d2_value)
        print(norm.cdf(d1_value), norm.cdf(d2_value))
        print(42 * norm.cdf(d1_value) - k_dis * norm.cdf(d2_value))

        d1_value: float = d1(100, 103, 0.05, 2, 0.5)
        d2_value: float = d2(d1_value, 2, 0.5)
        k_dis: float = 103 * np.exp(-1 * 0.05 * 0.5)
        print(d1_value, d2_value)
        print(norm.cdf(d1_value), norm.cdf(d2_value))
        print(100 * norm.cdf(d1_value) - k_dis * norm.cdf(d2_value))

    def test_bsm(self):
        # print(OptionsCalculator.black_scholes_pricing('c', 103, 0.5, 2, 100, 0.05))
        # print(OptionsCalculator.black_scholes_pricing('c', 40, 0.5, 0.2, 42, 0.1))

        # print(OptionsCalculator.black_scholes_pricing('c', 60, 5, 0.3, 40, 0.03))

        # print(OptionsCalculator.black_scholes_pricing(lambda: 50, 'p', 50, 0.25, 0.3,
        #                                              0.1))
        # print(OptionsCalculator.black_scholes_pricing(lambda: 50 - 1.50 * np.exp(-1 * 0.1 * (1/6)),
        #                                              'p', 50, 0.25, 0.3, 0.1))

        print(OptionsCalculator.black_scholes_merton(SecEnum.EQUITY)(lambda: 50, 'c', 50, 5, 0.25, 0.05))

        print(OptionsCalculator.black_scholes_merton(SecEnum.CURRENCY)(1.6, 'c', 1.6, 1 / 3, 0.141, 0.08, 0.11))

        parameters: dict = {
            'option_price': 2.50,
            'security_price_func': lambda: 15,
            'option_type': 'c',
            'strike': 13,
            'expiry': 0.25,
            'r': 0.05
        }
        result: RootResults = OptionsCalculator.bsm_implied_volatility(SecEnum.EQUITY, option_price=2.50,
                                                                       security_price_func=lambda: 15,
                                                                       option_type='c',
                                                                       strike=13,
                                                                       expiry=0.25,
                                                                       interest_rate=0.05)
        print('Exercise 13.16\n')
        print(result)
        print()
        result: RootResults = OptionsCalculator.bsm_implied_volatility(SecEnum.CURRENCY, option_price=0.043,
                                                                       exchange_rate=1.6,
                                                                       option_type='c',
                                                                       strike=1.6,
                                                                       expiry=1 / 3,
                                                                       interest_rate=0.08,
                                                                       foreign_interest_rate=0.11)
        print(result)
        print(result.root)

    def test_bsm_hull_chpter_15(self):
        # 15.6
        print(OptionsCalculator.black_scholes_merton(SecEnum.INDEX)(lambda: 250, 'c', 250, 0.25, 0.18, 0.1, 0.03))

        # 15.7
        print(OptionsCalculator.black_scholes_merton(SecEnum.CURRENCY)(0.52, 'p', 0.50, 8 / 12, 0.12, 0.04, 0.08))

        # 15.11
        print(OptionsCalculator.black_scholes_merton(SecEnum.INDEX)(lambda: 696, 'p', 700, 0.25, 0.30, 0.07, 0.04))

        # 15.16
        print(OptionsCalculator.black_scholes_merton(SecEnum.INDEX)(lambda: 1200, 'p', 1134, 1.0, 0.30, 0.07, 0.04))

        # 17.00 Section 17.1
        print(OptionsCalculator.black_scholes_merton(SecEnum.EQUITY)(lambda: 49, 'c', 50, 0.3846, 0.20, 0.05, 0.00))
        print(OptionsCalculator.black_scholes_merton(SecEnum.EQUITY)(lambda: 49, 'c', 50, 0.3846, 0.20, 0.05, 0.00) * 100000)
        print(OptionsCalculator.d1(48.12,50.0,0.05, 0.00,0.2, 0.36539))
        print(
            norm.cdf(OptionsCalculator.d1(48.12,50.0,0.05, 0.00,0.2, 0.36539))
        )

    def test_sample_mean(self):
        _1970_: () = (0.580, 4.598, 120)
        _1980_: () = (1.470, 4.738, 120)
        print(ttest_2_samples(_1970_, _1980_))

        utility: () = (64.42, 14.03, 21)
        non_utility: () = (55.75, 25.17, 64)
        print(ttest_2_samples(utility, non_utility))


    def test_payoff_function(self):
        mpl.rcParams['font.family'] = 'serif'
        k: int = 8000
        s = np.linspace(7000,9000, 100)
        h = np.maximum(k-s,0) - 30
        print(type(h))

        plt.figure()
        plt.plot(s,h,lw=2.5)
        plt.xlabel('index level $s_t$ at maturity')
        plt.ylabel('inner value of European call option')
        plt.grid(True)
        plt.show()

    def test_bsm_hull_chpter_17(self):

        # 17.00 Section 17.1
        print(OptionsCalculator.black_scholes_merton(SecEnum.EQUITY)(lambda: 49, 'c', 50, 0.3846, 0.20, 0.05, 0.00))
        print(OptionsCalculator.black_scholes_merton(SecEnum.EQUITY)(lambda: 49, 'c', 50, 0.3846, 0.20, 0.05, 0.00) * 100000)
        print(OptionsCalculator.d1(48.12,50.0,0.05, 0.00,0.2, 0.36539))
        print(
            norm.cdf(OptionsCalculator.d1(48.12,50.0,0.05, 0.00,0.2, 0.36539))
        )
        print(
            norm.cdf(OptionsCalculator.d1(47.37, 50.0, 0.05, 0.00, 0.2, 0.3462))
        )
        print(
            norm.cdf(OptionsCalculator.d1(57.25, 50.0, 0.05, 0.00, 0.2, 0.0001))
        )

    def test_hull_chapter_18(self):
        result: RootResults = OptionsCalculator.bsm_implied_volatility(SecEnum.CURRENCY, option_price=0.0236,
                                                                       exchange_rate=0.60,
                                                                       option_type='c',
                                                                       strike=0.59,
                                                                       expiry=1,
                                                                       interest_rate=0.05,
                                                                       foreign_interest_rate=0.1)
        print(result)
        print(result.root)

    def test_bond_pricing_vanilla(self):
        result: float = BondPricing.bond_pricing_vanilla(1000, 10, 0.08,
                                                         0.060)
        print(result)


    def test_bodie_page_461(self):
        bond1_price: float = BondPricing.bond_pricing_vanilla(1000, 10, 0.06,
                                                         0.060)
        bond2_price: float = BondPricing.bond_pricing_vanilla(1000, 10, 0.08,
                                                              0.060)
        print(bond1_price)
        print(bond2_price)

        # Concept Check 14.4
        bond3_price: float = BondPricing.bond_pricing_vanilla(1000, 20, 0.09,
                                                              0.080)
        def func(x: float) -> float:
            return bond3_price - BondPricing.bond_pricing_vanilla(1050, 5, 0.09, x)

        ytm = opt.bisect(func, 0.01, 0.09, full_output= True)
        print(bond3_price)
        print(ytm)
        from datetime import date
        print( type(date.today().isocalendar().year))

        current_year: int = date.today().isocalendar().year
        jan_cycle : date = date(year=current_year,month=1,day=1)
        feb_cycle : date = date(year=current_year,month=2,day=1)
        mar_cycle : date = date(year=current_year,month=3,day=1)

        print( jan_cycle)
        print( feb_cycle)
        print( mar_cycle)
        print( date(year=current_year,month=jan_cycle.month+3,day=1))

        list = [date(year=current_year,month=jan_cycle.month+i,day=1) for i in range(11) if i > 0 and i % 3 == 0 ]
        list.insert(0,jan_cycle)

        str_list = [i.strftime("%b").upper() for i in list]
        print(list)
        print(str_list)


