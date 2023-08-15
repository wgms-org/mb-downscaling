"""
Interpolate glacier mass balance from seasonal observations.
"""
import datetime
from typing import Dict, Iterable, Tuple

import pandas as pd
import numpy as np


def evaluate_sine(
    x: Iterable[float],
    a: float = 1,
    b: float = 1,
    c: float = 0,
    d: float = 0,
    mask: Iterable[float] = None
) -> np.ndarray:
    """
    Evaluate sine function.

    The function is assumed to be of the form:

    f(x) = a sin(b(x - c)) + d

    Arguments:
        x: Values at which to evaluate the function.
        a: Amplitude.
        b: Period scaling (period = 2π/b).
        c: Phase shift (positive: right).
        d: Vertical shift (positive: up).
        mask: Interval outside of which the function is zero.

    Returns:
        Function values.

    Examples:
        >>> evaluate_sine([0, np.pi/2, np.pi, 3 * np.pi/2, 2 * np.pi]).round(12)
        array([ 0.,  1.,  0., -1., -0.])
        >>> evaluate_sine([0, np.pi/2, np.pi], a=2).round(12)
        array([0., 2., 0.])
        >>> evaluate_sine([0, np.pi/2, np.pi], d=1)
        array([1., 2., 1.])
        >>> evaluate_sine([0, np.pi/2, np.pi], d=1, mask=[0, np.pi/2])
        array([0., 0., 1.])
    """
    x = np.atleast_1d(x)
    f = a * np.sin(b * (x - c)) + d
    if mask is not None:
        f[(x >= min(*mask)) & (x <= max(*mask))] = 0
    return f


def integrate_sine(
    intervals: Iterable[Iterable[float]],
    a: float = 1,
    b: float = 1,
    c: float = 0,
    d: float = 0,
    mask: Iterable[float] = None
) -> np.ndarray:
    """
    Integrate sine function.

    The function is assumed to be of the form:

    f(x) = a sin(b(x - c)) + d

    The integral in the interval [i, j] is then:

    a(cos(b(i - c)) - cos(b(j - c))) / b + d(j - i)

    Arguments:
        intervals: Intervals to integrate over [(i1, j1), ..., (in, jn)].
        a: Amplitude.
        b: Period scaling (period = 2π/b).
        c: Phase shift (positive: right).
        d: Vertical shift (positive: up).
        mask: Interval outside of which the integral is zero.

    Returns:
        Integral for each interval.

    Examples:
        >>> integrate_sine([(0, np.pi), (np.pi, 2 * np.pi), (0, 2 * np.pi)])
        array([ 2., -2.,  0.])
        >>> integrate_sine([(0, np.pi), (np.pi, 2 * np.pi)], a=2)
        array([ 4., -4.])
        >>> integrate_sine([(0, np.pi), (np.pi, 2 * np.pi)], d=2/np.pi)
        array([4., 0.])
        >>> integrate_sine([(0, np.pi/2), (np.pi/2, np.pi)], mask=[0, np.pi/2])
        array([1., 0.])
    """
    intervals = np.atleast_2d(intervals)
    if mask is not None:
        # Restrict intervals to mask
        intervals = intervals.copy()
        xmin, xmax = min(*mask), max(*mask)
        intervals[intervals < xmin] = xmin
        intervals[intervals > xmax] = xmax
    i = intervals[:, 0]
    j = intervals[:, 1]
    return a * (np.cos(b * (i - c)) - np.cos(b * (j - c))) / b + d * (j - i)


def generate_seasonal_sine(
    balance: float,
    interval: Iterable[float] = [0, 0.5],
    annual_balance: float = 0
) -> Dict[str, float]:
    """
    Generate a sine function that estimates a mass balance season.

    Arguments:
        balance: Seasonal balance.
        interval: Seasonal interval, assuming the hydrological year is [0, 1].
        annual_balance: Annual balance applied uniformally over the year.

    Returns:
        Sine function parameters (see `evaluate_sine`, `integrate_sine`).

    Examples:
        >>> intervals = [(0, 0.75), (0.75, 1)]

        Summer and winter seasonal balances.

        >>> Bw, Bs = 4, -2
        >>> sine = generate_seasonal_sine(Bw, interval=intervals[0])
        >>> integrate_sine(intervals[0], **sine)[0] == Bw
        True
        >>> sine = generate_seasonal_sine(Bs, interval=intervals[1])
        >>> integrate_sine(intervals[1], **sine)[0] == Bs
        True
        >>>

        Annual balance and balance amplitude with uniform annual balance.

        >>> Ba, alpha = 2, 3
        >>> Bwi, Bsi = Ba/2 + alpha, Ba/2 - alpha
        >>> sine = generate_seasonal_sine(
        ...     Bwi, interval=intervals[0], annual_balance=Ba
        ... )
        >>> Bw = integrate_sine(intervals[0], **sine)[0]
        >>> sine = generate_seasonal_sine(
        ...     Bsi, interval=intervals[1], annual_balance=Ba
        ... )
        >>> Bs = integrate_sine(intervals[1], **sine)[0]
        >>> Ba == Bw + Bs
        True
        >>> alpha == abs(Bw - Bs) / 2
        True
    """
    width = abs(interval[0] - interval[1])
    b = (2 * np.pi) / (2 * width)
    return {
        'a': (balance - annual_balance * width) * b / 2,
        'b': b,
        'c': min(*interval),
        'd': annual_balance
    }


def seasonal_balances_from_annual_balance(
    annual_balance: float,
    balance_amplitude: float
) -> Tuple[float, float]:
    """
    Calculate seasonal balances from annual balance and amplitude.

    Returns:
        Winter and summer balances.
    """
    return (
        annual_balance / 2 + balance_amplitude,
        annual_balance / 2 - balance_amplitude
    )


def sine_interpolation_from_mean_balances(
    balance_amplitude: float,
    annual_balance: float,
    temporal_resolution: int = 365,
    winter_fraction: float = 0.5,
    uniform_annual_balance: bool = False
) -> pd.DataFrame:
    """
    Interpolate mass balance for a full year from annual balance and amplitude.

    Winter and summer balances are each represented by their own sine function
    (see `generate_seasonal_sine`).

    Arguments:
        balance_amplitude: Mass-balance amplitude
            (see `calc_mass_balance_amplitude`).
        annual_balance: Annual mass balance.
        temporal_resolution: Temporal resolution of output,
            e.g. 12 (~monthly) or 365 (~daily).
        winter_fraction: Annual fraction of winter season.
        uniform_annual_balance: Whether annual balance should be applied
            uniformally over the year (True) rather than assume that all mass
            gain/loss occurs in winter/summer (False).

    Returns:
        Dataframe with columns TIME_STEP (float) and BALANCE (float).

    Examples:
        >>> balance_amplitude = 5
        >>> annual_balance = -3

        With winter and summer of equal length:

        >>> balances = sine_interpolation_from_mean_balances(
        ...     balance_amplitude, annual_balance, temporal_resolution=12
        ... )['BALANCE']
        >>> np.isclose(annual_balance, balances.sum())
        True
        >>> np.isclose(
        ...     balance_amplitude,
        ...     abs(balances[:6].sum() - balances[6:].sum()) / 2
        ... )
        True

        With winter and summer of different lengths:

        >>> balances = sine_interpolation_from_mean_balances(
        ...     balance_amplitude, annual_balance, temporal_resolution=12,
        ...     winter_fraction=0.75
        ... )['BALANCE']
        >>> np.isclose(annual_balance, balances.sum())
        True
        >>> np.isclose(
        ...     balance_amplitude,
        ...     abs(balances[:9].sum() - balances[9:].sum()) / 2
        ... )
        True

        And with uniform annual balance:

        >>> balances = sine_interpolation_from_mean_balances(
        ...     balance_amplitude, annual_balance, temporal_resolution=12,
        ...     winter_fraction=0.75, uniform_annual_balance=True
        ... )['BALANCE']
        >>> np.isclose(annual_balance, balances.sum())
        True
        >>> np.isclose(
        ...     balance_amplitude,
        ...     abs(balances[:9].sum() - balances[9:].sum()) / 2
        ... )
        True
    """
    # Calculate seasonal balances from annual balance and balance amplitude
    winter_balance, summer_balance = seasonal_balances_from_annual_balance(
        annual_balance=annual_balance,
        balance_amplitude=balance_amplitude
    )
    if uniform_annual_balance:
        edges = np.linspace(0, 1, num=temporal_resolution + 1)
        intervals = np.column_stack((edges[:-1], edges[1:]))
        sine_w = generate_seasonal_sine(
            winter_balance,
            interval=[0, winter_fraction],
            annual_balance=annual_balance
        )
        sine_s = generate_seasonal_sine(
            summer_balance,
            interval=[winter_fraction, 1],
            annual_balance=annual_balance
        )
        balances = (
            integrate_sine(intervals, **sine_w, mask=[0, winter_fraction]) +
            integrate_sine(intervals, **sine_s, mask=[winter_fraction, 1])
        )
        t = np.arange(0.5, temporal_resolution, 1)
        return pd.DataFrame({'TIME_STEP': t, 'BALANCE': balances})
    return sine_interpolation_from_seasonal_balances(
        winter_balance=winter_balance,
        summer_balance=summer_balance,
        temporal_resolution=temporal_resolution,
        winter_fraction=winter_fraction
    )


def sine_interpolation_from_seasonal_balances(
    winter_balance: float,
    summer_balance: float,
    temporal_resolution: int = 365,
    winter_fraction: float = 0.5
) -> pd.DataFrame:
    """
    Interpolate mass balance for a full year from seasonal balances.

    Winter and summer balances are each represented by their own sine function
    (see `generate_seasonal_sine`).

    Arguments:
        winter_balance: Winter mass balance.
        summer_balance: Summer mass balance.
        temporal_resolution: Temporal resolution of output,
            e.g. 12 (~monthly) or 365 (~daily).
        winter_fraction: Annual fraction of winter season.

    Returns:
        Dataframe with columns TIME_STEP (float) and BALANCE (float).

    Examples:
        >>> winter_balance = 4
        >>> summer_balance = -3

        Single sample equals annual balance.

        >>> balances = sine_interpolation_from_seasonal_balances(
        ...     winter_balance=winter_balance, summer_balance=summer_balance,
        ...     temporal_resolution=1, winter_fraction=0.5
        ... )['BALANCE']
        >>> np.isclose(balances[0], winter_balance + summer_balance)
        True

        Two samples with equal length season equal seasonal balances.

        >>> balances = sine_interpolation_from_seasonal_balances(
        ...     winter_balance=winter_balance, summer_balance=summer_balance,
        ...     temporal_resolution=2, winter_fraction=0.5
        ... )['BALANCE']
        >>> np.isclose(balances[0], winter_balance)
        True
        >>> np.isclose(balances[1], summer_balance)
        True

        Samples with one spanning both seasons still sum to annual balance.

        >>> balances = sine_interpolation_from_seasonal_balances(
        ...     winter_balance=winter_balance, summer_balance=summer_balance,
        ...     temporal_resolution=3, winter_fraction=0.5
        ... )['BALANCE']
        >>> np.isclose(balances.sum(), winter_balance + summer_balance)
        True
    """
    edges = np.linspace(0, 1, num=temporal_resolution + 1)
    intervals = np.column_stack((edges[:-1], edges[1:]))
    sine_w = generate_seasonal_sine(
        winter_balance, interval=[0, winter_fraction]
    )
    sine_s = generate_seasonal_sine(
        summer_balance, interval=[winter_fraction, 1]
    )
    balances = (
        integrate_sine(intervals, **sine_w, mask=[0, winter_fraction]) +
        integrate_sine(intervals, **sine_s, mask=[winter_fraction, 1])
    )
    t = np.arange(0.5, temporal_resolution, 1)
    return pd.DataFrame({'TIME_STEP': t, 'BALANCE': balances})


def interpolate_daily_balances(
    bwsa_df: pd.DataFrame,
    alpha: float = None,
    winter_fraction: float = 0.5,
    winter_start: Tuple[bool, int, int] = (False, 10, 1),
    uniform_annual_balance: bool = False
) -> pd.DataFrame:
    """
    Interpolate daily mass balance from either seasonal or annual balances.

    See `sine_interpolation_from_seasonal_balances` and
    `sine_interpolation_from_mean_balances` for the two strategies used.

    Arguments:
        bwsa_df: DataFrame with columns
            Year (int), WINTER_BALANCE (float), SUMMER_BALANCE (float),
            and ANNUAL_BALANCE (float).
        alpha: Mass-balance amplitude. If not provided,
            it is computed from the seasonal balances in `bwsa_df`
            (see `calc_mass_balance_amplitude`).
        winter_fraction: Annual fraction of winter season.
        winter_start: Date of the start of winter as a year offset
            (False: previous, True: current), month, and day.
        uniform_annual_balance: Whether annual balance should be applied
            uniformally over the year (True) rather than assume that all mass
            gain/loss occurs in winter/summer (False).

    Returns:
        Dataframe with index DATE (datetime) and column BALANCE (float).

    Raises:
        ValueError: If winter_start is February 29.
    """
    is_start_year, month, day = winter_start
    if alpha is None:
        alpha = calc_mass_balance_amplitude(bwsa_df)
    if month == 2 and day == 29:
        raise ValueError('Winter cannot start on a leap day (February 29)')
    years = []
    for row in bwsa_df.to_dict(orient='records'):
        # Create series of every date in the hydrological year
        start_year = int(row['Year']) - (0 if is_start_year else 1)
        start_date = datetime.datetime(start_year, month, day)
        end_date = (
            start_date.replace(year=start_year + 1) - datetime.timedelta(days=1)
        )
        dates = pd.date_range(start_date, end_date, freq='D')
        n_dates = dates.size
        # Interpolate daily balances
        if np.isnan(row['WINTER_BALANCE']) or np.isnan(row['SUMMER_BALANCE']):
            # Use annual balance
            balances = sine_interpolation_from_mean_balances(
                annual_balance=row['ANNUAL_BALANCE'],
                balance_amplitude=alpha,
                winter_fraction=winter_fraction,
                temporal_resolution=n_dates,
                uniform_annual_balance=uniform_annual_balance
            )
        else:
            # Use seasonal balances
            balances = sine_interpolation_from_seasonal_balances(
                winter_balance=row['WINTER_BALANCE'],
                summer_balance=row['SUMMER_BALANCE'],
                winter_fraction=winter_fraction,
                temporal_resolution=n_dates
            )
        year = pd.DataFrame({'BALANCE': balances['BALANCE'], 'DATE': dates})
        years.append(year)
    return pd.concat(years).set_index('DATE')


def calc_mass_balance_amplitude(bwsa_df: pd.DataFrame) -> float:
    """
    Calculate the mean mass-balance amplitude from winter and summer balances.

    Mass-balance amplitude is defined as half the absolute value of the
    difference between the summer and winter balance.

    The term and definiton are based on
    Cogley et al. 2011 (https://unesdoc.unesco.org/ark:/48223/pf0000192525) and
    Braithwaite & Hughes 2020 (https://doi.org/10.3389/feart.2020.00302).

    Arguments:
        bwsa_df: Dataframe with columns WINTER_BALANCE (float) and
            SUMMER_BALANCE (float).

    Returns:
        Mean of mass-balance amplitudes.
    """
    alphas = ((bwsa_df['WINTER_BALANCE'] - bwsa_df['SUMMER_BALANCE']) / 2).abs()
    return alphas.mean()
