import calendar
from datetime import date


def actual_actual(settlement_date: date, coupon_dates: list) -> ():
    num_of_days_between: int = (settlement_date - coupon_dates[0]).days
    num_of_days_in_period: int = (coupon_dates[1] - coupon_dates[0]).days
    result = (num_of_days_between, num_of_days_in_period, num_of_days_between / num_of_days_in_period)

    return result


def thirty_three_sixty(settlement_date: date, coupon_dates: list) -> ():
    # Retrieve the integer value on months
    settlement_month: int = settlement_date.month
    first_coupon_month: int = coupon_dates[0].month
    second_coupon_month: int = coupon_dates[1].month

    # Get the last day of the first coupon month, i.e. the number of days in that month
    day_value: int = calendar.monthrange(coupon_dates[0].year, coupon_dates[0].month)[1]

    # Create the date object representing the last date of the first coupon payment month
    last_date_first_coupon_month: date = date(coupon_dates[0].year, coupon_dates[0].month, day_value)

    # create the first day of the second coupon payment
    first_date_second_coupon_month: date = date(coupon_dates[1].year, coupon_dates[1].month, 1)

    #perform the calculation based 30/360
    # i.e.
    # number of months between * 30 + the remainder days of the month
    remaining_first_coupon_days: int = (last_date_first_coupon_month - coupon_dates[0]).days
    num_of_days_between: int = (((settlement_month - first_coupon_month - 1) * 30) + remaining_first_coupon_days)

    # This value represents the days before the second payment date within the same month
    # i.e.,
    # second_coupon_payment date - first date of the same month
    days_before_second_coupon: int = (coupon_dates[1] - first_date_second_coupon_month).days
    num_of_days_in_period: int = (((second_coupon_month - first_coupon_month - 1) * 30) +
                                  remaining_first_coupon_days + days_before_second_coupon)

    result = (num_of_days_between, num_of_days_in_period, num_of_days_between / num_of_days_in_period)
    return result


def actual_three_sixty(settlement_date: date, coupon_dates: list) -> ():
    pass
