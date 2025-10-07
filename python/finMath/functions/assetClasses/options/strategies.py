from functions.assetClasses.options import OptionContract, option_payoff


def bull_spread(buy: OptionContract, sell: OptionContract, asset_price: float) -> ():

        buy_type:str = buy.get_contract_specs().get_option_type()
        sell_type:str = sell.get_contract_specs().get_option_type()

        buy_pay_off: float = float('-inf')
        sell_pay_off: float = float('-inf')

        if buy_type == sell_type :
                buy_pay_off:float = option_payoff(buy.get_contract_specs().get_strike(),
                                                  asset_price, buy_type).get_intrinsic_value()
                sell_pay_off: float = option_payoff(sell.get_contract_specs().get_strike(),
                                           asset_price, sell_type).get_intrinsic_value()

        return buy_pay_off, sell_pay_off, buy_pay_off - sell_pay_off