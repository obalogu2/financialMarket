from enum import Enum, IntEnum, StrEnum

import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar, RootResults

class OptionPayOffResult:
    def __init__(self, pay_off_value: float, intrinsic_value: float):
        self.__pay_off_value__ : float = pay_off_value
        self.__intrinsic_value__: float = intrinsic_value

    def get_pay_off_value(self) -> float:
        return self.__pay_off_value__

    def get_intrinsic_value(self) -> float:
        return self.__intrinsic_value__

    def __str__(self):
        return f"(PayOff,Intrinsic)=({self.__pay_off_value__},{self.__intrinsic_value__})"

def option_payoff(strike: float, underlying: float, option_type: str) -> OptionPayOffResult:
    payoff_table = {'C': lambda s, k: OptionPayOffResult(s - k, max(s - k, 0)),
                    'P': lambda s, k: OptionPayOffResult(k - s, max(k - s, 0))}
    return payoff_table[option_type](underlying, strike)

class OptionType(StrEnum):
    CALL: str = 'C'
    PUT: str = 'P'

class OptionStyle(StrEnum):
    AMER: str = 'A'
    EURO: str = 'E'

class OptionClass:
    def __init__(self, underlying: str, option_type: OptionType):
        self.__underlying__: str = underlying
        self.__option_type__: OptionType = option_type
        self.__strikes__: set = set()

    def get_underlying(self) -> str:
        return self.__underlying__

    def get_option_type(self) -> OptionType:
        return self.__option_type__

    def add_strike(self, strike: float):
        self.__strikes__.add(strike)

class ContractSpecs:

    """
    This class is a Record Holder specifying the option contract specification.

    """
    def __init__(self, strike: float, option_type: OptionType = OptionType.CALL,
                 option_style: OptionStyle =  OptionStyle.EURO, delivery: int = 100,
                 expiry: float = None):
        self.__option_type: str = option_type
        self.__strike: float = strike
        self.__delivery: int = delivery
        self.__expiry: float = expiry
        self.__option_style: str = option_style

    def get_option_type(self) -> str:
        return self.__option_type

    def get_strike(self) -> float:
        return self.__strike

    def get_delivery(self) -> int:
        return self.__delivery

    def get_expiry(self) -> float:
        return self.__expiry

    def get_option_style(self) -> str:
        return self.__option_style


class OptionContract:
    def __init__(self, underlying: float, option_price: float, contract_specs: ContractSpecs):
        self.__underlying: float = underlying
        self.__contract_specs: ContractSpecs = contract_specs
        self.__option_price: float = option_price

    def get_underlying(self) -> float:
        return self.__underlying

    def get_option_price(self) -> float:
        return self.__option_price

    def get_contract_specs(self) -> ContractSpecs:
        return self.__contract_specs


class SecEnum(IntEnum):
    EQUITY = 1
    FUTURES = 2
    CURRENCY = 3
    INDEX = 4

    def __add__(self, other) -> int:
        raise Exception('Unsupported operand type')


class OptionsCalculator:

    @staticmethod
    def naked_option_initial_margin(option_contract: OptionContract, quantity: int) -> float:
        total_quantity: int = option_contract.get_contract_specs().get_delivery() * quantity
        option_type: str = option_contract.get_contract_specs().get_option_type()
        option_strike: float = option_contract.get_contract_specs().get_strike()
        underlying_price: float = option_contract.get_underlying()

        pay_off: float = option_payoff(underlying_price, option_strike, option_type).get_pay_off_value()
        out_of_the_money: float = 0.00
        if pay_off < 0:
            out_of_the_money = abs(pay_off)

        val1: float = option_contract.get_option_price() + 0.2 * option_contract.get_underlying() - out_of_the_money
        val2: float = option_contract.get_option_price() + 0.1 * option_contract.get_underlying()

        return total_quantity * max(val1, val2)

    @staticmethod
    def __a(underlying_type: str, expiry_factor: float):

        if underlying_type == 'STK' or underlying_type == 'INDX':
            return lambda r, q=0: np.exp((r - q) * expiry_factor)
        elif underlying_type == 'CURR':
            return lambda r, rf: np.exp((r - rf) * expiry_factor)
        elif underlying_type == 'FUT':
            return lambda: 1
        else:
            return lambda *args: 0

    @staticmethod
    def risk_neutral_probability(underlying_type: str, time_step: int, expiry: float, volatility: float,
                                 *args) -> float:
        expiry_factor: float = expiry / time_step
        u: float = np.exp(volatility * np.sqrt(expiry_factor))
        d: float = 1 / u
        a: float = OptionsCalculator.__a(underlying_type, expiry_factor)(*args)

        return (a - d) / (u - d)

    @staticmethod
    def options_multiplier(current_price: float, up_price: float, down_price: float, interest_rate: float,
                           expiry: float, time_step: int, discounting: bool = False) -> dict:
        u: float = up_price / current_price
        d: float = down_price / current_price
        a: float = np.exp(-1 if discounting else 1 * interest_rate * (expiry / time_step))
        p: float = (a - d) / (u - d)

        return {'u': u, 'd': d, 'a': a, 'p': p}

    @staticmethod
    def d1(current_price: float, strike: float, interest_rate: float, dividend_yield: float, volatility: float,
           expiry: float) -> float:
        return ((np.log(current_price / strike) + (
                interest_rate - dividend_yield + (volatility ** 2) / 2) * expiry) /
                (volatility * np.sqrt(expiry)))

    @staticmethod
    def d2(d1_value: float, volatility: float, expiry: float) -> float:
        return d1_value - (volatility * np.sqrt(expiry))

    @staticmethod
    def black_scholes_merton(security_enum: SecEnum):
        match security_enum:
            case SecEnum.EQUITY | SecEnum.INDEX:
                return OptionsCalculator.__black_scholes_pricing_equity__
            case SecEnum.CURRENCY:
                return OptionsCalculator.__black_scholes_pricing_currency__

    @staticmethod
    def __black_scholes_pricing_equity__(security_price_func, option_type: str, strike: float, expiry: float,
                                         volatility: float,
                                         interest_rate: float, dividend_yield: float = 0.00) -> float:

        current_price: float = security_price_func()

        def d1() -> float:
            return ((np.log(current_price / strike) + (
                    interest_rate - dividend_yield + (volatility ** 2) / 2) * expiry) /
                    (volatility * np.sqrt(expiry)))

        def d2() -> float:
            return ((np.log(current_price / strike) + (
                    interest_rate - dividend_yield - (volatility ** 2) / 2) * expiry) /
                    (volatility * np.sqrt(expiry)))

        # discounting factor
        def df(v: float, k: float, e: float) -> float:
            return v * np.exp(-1 * k * e)

        if option_type == 'c':
            return df(current_price, dividend_yield, expiry) * norm.cdf(d1()) - df(strike, interest_rate, expiry) * norm.cdf(d2())
        else:
            return df(strike, interest_rate, expiry) * norm.cdf(-1 * d2()) - df(current_price, dividend_yield, expiry) * norm.cdf(-1 * d1())

    @staticmethod
    def __black_scholes_pricing_currency__(rate: float, option_type: str, strike: float, expiry: float,
                                           volatility: float, int_rate: float,
                                           foreign_int_rate: float, rate_type: str = 'spot') -> float:

        def d1(r: float, rf: float) -> float:
            return ((np.log(rate / strike) + (
                    r - rf + (volatility ** 2) / 2) * expiry) /
                    (volatility * np.sqrt(expiry)))

        def d2(r: float, rf: float) -> float:
            return ((np.log(rate / strike) + (
                    r - rf - (volatility ** 2) / 2) * expiry) /
                    (volatility * np.sqrt(expiry)))

        def k_disc() -> float:
            return strike * np.exp(-1 * int_rate * expiry)

        rate_discounting: dict = {'spot': lambda: rate * np.exp(-1 * foreign_int_rate * expiry),
                                  'forward': lambda: rate * np.exp(-1 * int_rate * expiry)}
        d1_d2_args: dict = {'spot': (int_rate, foreign_int_rate), 'forward': (0, 0)}

        if option_type == 'c':
            return (rate_discounting[rate_type]() * norm.cdf(d1(d1_d2_args[rate_type][0], d1_d2_args[rate_type][1]))
                    - k_disc() * norm.cdf(d2(d1_d2_args[rate_type][0], d1_d2_args[rate_type][1])))
        else:
            return (k_disc() * norm.cdf(-1 * d2(d1_d2_args[rate_type][0], d1_d2_args[rate_type][1]))
                    - rate_discounting[rate_type]() * norm.cdf(
                        -1 * d1(d1_d2_args[rate_type][0], d1_d2_args[rate_type][1])))

    @staticmethod
    def bsm_implied_volatility(security_enum: SecEnum, root_finder_func='secant', **kwargs) -> RootResults:

        def blm_volatility_wrapper(volatility) -> float:

            match security_enum:
                case SecEnum.EQUITY:
                    if not ('dividend_yield' in kwargs.keys()):
                        kwargs['dividend_yield'] = 0.0

                    return kwargs['option_price'] - OptionsCalculator.__black_scholes_pricing_equity__(
                        kwargs['security_price_func'],
                        kwargs['option_type'],
                        kwargs['strike'],
                        kwargs['expiry'], volatility,
                        kwargs['interest_rate'],
                        kwargs['dividend_yield'])

                case SecEnum.CURRENCY:

                    if not ('foreign_interest_rate' in kwargs.keys()):
                        kwargs['foreign_interest_rate'] = 0.0

                    return kwargs['option_price'] - OptionsCalculator.__black_scholes_pricing_currency__(
                        kwargs['exchange_rate'],
                        kwargs['option_type'],
                        kwargs['strike'],
                        kwargs['expiry'], volatility,
                        kwargs['interest_rate'],
                        kwargs['foreign_interest_rate'])
                case _:
                    raise Exception('Only the following options are supported:' + str(SecEnum))

        return root_scalar(blm_volatility_wrapper, method=root_finder_func, x0=0.001, x1=1)
