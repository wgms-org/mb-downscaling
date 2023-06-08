"""
Interpolate glacier mass balance from seasonal observations.
"""
import datetime
from typing import Tuple

import pandas as pd
import numpy as np


def sine_interpolation_from_mean_balances(
    annual_balance: float,
    balance_amplitude: float,
    winter_fraction: float = 0.5,
    temporal_resolution: int = 365,
    uniform_annual_balance: bool = True
) -> pd.DataFrame:
    """
    Interpolate mass balance for a full year from annual balance and amplitude.

    Winter and summer balances are each represented by their own sine function
    of the form:

    f(t) = A sin(Bt) + D,

    where t is time and A, B, and D are parameters chosen to match
    `winter_fraction`, `annual_balance`, and `balance_amplitude`.

    Arguments:
        annual_balance: Annual mass balance.
        balance_amplitude: Mass-balance amplitude
            (see `calc_mass_balance_amplitude`).
        winter_fraction: Annual fraction of winter season.
        temporal_resolution: Temporal resolution of output,
            e.g. 12 (~monthly) or 365 (~daily).
        uniform_annual_balance: Whether annual balance should be applied
            uniformally over the year (True) rather than assume that all mass
            gain/loss occurs in winter/summer (False).

    Returns:
        Dataframe with columns TIME_STEP (float) and BALANCE (float).

    Examples:
        >>> annual_balance = -3
        >>> balance_amplitude = 5

        With winter and summer of equal length:

        >>> balances = sine_interpolation_from_mean_balances(
        ...     annual_balance, balance_amplitude
        ... )['BALANCE']
        >>> np.isclose(annual_balance, balances.sum(), atol=1e-5)
        True
        >>> np.isclose(
        ...     balance_amplitude,
        ...     abs(balances[:182].sum() - balances[182:].sum()) / 2,
        ...     atol=1e-4
        ... )
        True

        With winter and summer of different lengths:

        >>> balances = sine_interpolation_from_mean_balances(
        ...     annual_balance, balance_amplitude, winter_fraction=0.75
        ... )['BALANCE']
        >>> np.isclose(annual_balance, balances.sum(), atol=1e-3)
        True
        >>> np.isclose(
        ...     balance_amplitude,
        ...     abs(balances[:273].sum() - balances[273:].sum()) / 2,
        ...     atol=1e-3
        ... )
        True

        And with non-uniform annual balance:

        >>> balances = sine_interpolation_from_mean_balances(
        ...     annual_balance, balance_amplitude, winter_fraction=0.75,
        ...     uniform_annual_balance=True
        ... )['BALANCE']
        >>> np.isclose(annual_balance, balances.sum(), atol=1e-3)
        True
        >>> np.isclose(
        ...     balance_amplitude,
        ...     abs(balances[:273].sum() - balances[273:].sum()) / 2,
        ...     atol=1e-3
        ... )
        True
    """
    # Set sine periods from winter fraction, with year in interval [0, 1]
    b_w = (2 * np.pi) / (temporal_resolution * 2 * winter_fraction)
    b_s = (2 * np.pi) / (temporal_resolution * 2 * (1 - winter_fraction))
    # Calculate seasonal balances from annual balance and balance amplitude
    Bs = annual_balance / 2 - balance_amplitude
    Bw = annual_balance - Bs
    # Compute sine amplitudes
    # seasonal balance = a * 2 / b
    a_w = Bw * b_w / 2
    a_s = Bs * b_s / 2
    # Adjust amplitudes for constant annual balance
    d = 0
    end_winter = temporal_resolution * winter_fraction
    if uniform_annual_balance:
        d = annual_balance / temporal_resolution
        # Seasonal balance changes due to shift
        Bw_a = d * end_winter
        Bs_a = annual_balance - Bw_a
        # Adjust sine amplitudes to compensate for change in seasonal balance
        a_w -= Bw_a * b_w / 2
        a_s -= Bs_a * b_s / 2
    # Compute balance at each time step
    t = np.arange(0.5, temporal_resolution, 1)
    is_winter = t < end_winter
    balances = np.concatenate((
        a_w * np.sin(b_w * t[is_winter]) + d,
        a_s * np.sin(b_s * (t[~is_winter] - end_winter)) + d,
    ))
    return pd.DataFrame({'TIME_STEP': t, 'BALANCE': balances})


def sine_interpolation_from_seasonal_balances(
    winter_balance: float,
    summer_balance: float,
    winter_fraction: float = 0.5,
    temporal_resolution: int = 365
) -> pd.DataFrame:
    """
    Interpolate mass balance for a full year from seasonal balances.

    Winter and summer balances are each represented by their own sine function
    of the form:

    f(t) = A sin(Bt) + D,

    where t is time and A, B, and D are parameters chosen to match
    `winter_fraction`, `winter_balance`, and `summer_balance`.

    Arguments:
        winter_balance: Winter mass balance.
        summer_balance: Summer mass balance.
        winter_fraction: Annual fraction of winter season.
        temporal_resolution: Temporal resolution of output,
            e.g. 12 (~monthly) or 365 (~daily).

    Returns:
        Dataframe with columns TIME_STEP (float) and BALANCE (float).
    """
    # Set sine periods from winter fraction
    b_w = (2 * np.pi) / (temporal_resolution * 2 * winter_fraction)
    b_s = (2 * np.pi) / (temporal_resolution * 2 * (1 - winter_fraction))
    # Compute sine amplitudes
    # seasonal balance = a * 2 / b
    a_w = winter_balance * b_w / 2
    a_s = summer_balance * b_s / 2
    # Compute balance at each time step
    end_winter = temporal_resolution * winter_fraction
    t = np.arange(0.5, temporal_resolution, 1)
    is_winter = t < end_winter
    balances = np.concatenate((
        a_w * np.sin(b_w * t[is_winter]),
        a_s * np.sin(b_s * (t[~is_winter] - end_winter)),
    ))
    return pd.DataFrame({'TIME_STEP': t, 'BALANCE': balances})


def interpolate_daily_balances(
    bwsa_df: pd.DataFrame,
    balance_amplitude: float = None,
    winter_fraction: float = 0.5,
    winter_start: Tuple[bool, int, int] = (False, 10, 1),
    uniform_annual_balance: bool = True
) -> pd.DataFrame:
    """
    Interpolate daily mass balance from either seasonal or annual balances.

    See `sine_interpolation_from_seasonal_balances` and
    `sine_interpolation_from_mean_balances` for the two strategies used.

    Arguments:
        bwsa_df: DataFrame with columns
            Year (int), WINTER_BALANCE (float), SUMMER_BALANCE (float),
            and ANNUAL_BALANCE (float).
        balance_amplitude: Mass-balance amplitude. If not provided,
            it is computed from the seasonal balances in `bwsa_df`
            (see `calc_mass_balance_amplitude`).
        winter_fraction: Annual fraction of winter season.
        winter_start: Date of the start of winter as a year offset
            (False: previous, True: current), month, and day.
        uniform_annual_balance: See Whether annual balance should be applied
            uniformally over the year (True) rather than assume that all mass
            gain/loss occurs in winter/summer (False).

    Returns:
        Dataframe with index DATE (datetime) and column BALANCE (float).
    """
    is_start_year, month, day = winter_start
    if balance_amplitude is None:
        balance_amplitude = calc_mass_balance_amplitude(bwsa_df)
    years = []
    for row in bwsa_df.to_dict(orient='records'):
        # Create series of every date in the hydrological year
        start_year = int(row['Year']) - (0 if is_start_year else 1)
        start_date = datetime.datetime(start_year, month, day)
        end_date = (start_date - datetime.timedelta(days=1)).replace(
            year=start_year + 1
        )
        dates = pd.date_range(start_date, end_date, freq='D')
        n_dates = dates.size
        # Interpolate daily balances
        if np.isnan(row['WINTER_BALANCE']) or np.isnan(row['SUMMER_BALANCE']):
            # Use annual balance
            balances = sine_interpolation_from_mean_balances(
                annual_balance=row['ANNUAL_BALANCE'],
                balance_amplitude=balance_amplitude,
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
