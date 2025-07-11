import unittest
import pandas as pd

from functions import RatesUtil
from functions.assetClasses.bond import BondPricing


class Chapter4Solution(unittest.TestCase):
    def test_Question_4_1(self):
        print(RatesUtil.conversion_btw_discrete_continuous(0.14, 4, False))
        print(RatesUtil.conversion_btw_discrete(0.10, 1, 2))

    def test_bond_pricing(self):
        df = pd.DataFrame({'maturity': [0.5, 1.0, 1.5], 'zero_rate': [10.0, 10.0, 6.76]})
        p: float = BondPricing.bond_pricing(df, 8.0, 100, bond_yield=10.4,
                                            first_maturity=0.5, n=3, difference=0.5)
        print(df.shape[0])

    def test_Question_4_3(self):
        result: float = BondPricing.bond_pricing_vanilla(100, 1.5, 0.08,
                                                         0.104)
        print(result)



if __name__ == '__main__':
    unittest.main()
