# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import multiprocessing as mp
import numpy as np
import pandas as pd
from functions import apply_interest,equivalent_rate


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def howmany_within_range(row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    print("number of processors: ", mp.cpu_count())
    # Prepare data
    np.random.RandomState(100)
    arr = np.random.randint(0, 10, size=[200000, 5])
    data = arr.tolist()
    print(data[:5])
    print(arr.size)

    results = []
    for row in data:
        results.append(howmany_within_range(row, minimum=4, maximum=8))

    print(results[:10])
    print(apply_interest(2, 0.02, 34, False))
    print(equivalent_rate(0.07, 1, True))

    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4], 'col3': [3, 4]})
    print(df.shape)
    print(df.shape[2:])
    print(df)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
