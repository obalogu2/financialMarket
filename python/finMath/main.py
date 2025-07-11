# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import multiprocessing as mp
from functions import apply_interest, RatesUtil


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print("number of processors: ", mp.cpu_count())

    print(apply_interest(2, 0.02, 34, False))
    print(RatesUtil.conversion_btw_discrete_continuous(0.14, 4, False))
    print(RatesUtil.conversion_btw_discrete(0.14, 4, 1))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
