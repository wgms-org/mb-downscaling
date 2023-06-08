"""
Interpolate glacier mass balance from seasonal observations.
"""
import pandas as pd
import numpy as np


def sine_interpolation_from_mean_balances(
    annual_balance: float,
    balance_amplitude: float,
    winter_fraction: float = 0.5,
    temporal_resolution: int = 364,
    uniform_annual_balance: bool = True
) -> pd.DataFrame:
    """
    Interpolate mass balance over a full year (winter-summer cycle).

    Winter and summer balances are each represented by their own sine function
    of the form:

    f(t) = A sin(Bt) + D,

    where t is time and A, B, and D are parameters chosen such that
    winter and summer have the right length and the resulting balances match
    (within rounding error) `annual_balance` and `balance_amplitude`.

    Arguments:
        annual_balance: Annual mass balance.
        balance_amplitude: Mass-balance amplitude
            (see `calc_mass_balance_amplitude`).
        winter_fraction: Annual fraction of winter season.
        temporal_resolution: Temporal resolution of output,
            e.g. 12 (~monthly) or 364 (~daily).
        uniform_annual_balance: Whether annual balance should be applied
            uniformally over the year (True) rather than assume that all mass
            gain/loss occurs in winter/summer.

    Returns:
        Interpolated mass balances.

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
    temporal_resolution: int = 364
) -> pd.DataFrame:
    """Sine function to interpolate mass balance over the year.
       The function allows to set different lengths of winter/summer seasons.
       Returns balances at sub-seasonal rates.

    Parameters:
        winter_balance:Winter balance, for a given year or mean over a given period.
        summer_balance:Summer balance, for a given year or mean over a given period.
        temporal_resolution:Temporal resolution as desired for output, e.g. 12 (monthly) or 364 (daily).
        winter_fraction: annual fraction of winter season, e.g. 8/12.

    Returns:
        sub_annual_balances_df:Dataframe with columns TIME_STEP (float) and BALANCE (float).
    """
    # Define sine functions parameters following the basic equation: y = f(x) = A sin(Bx + C) + D
    b_w = (2 * np.pi) / (temporal_resolution * 2 * winter_fraction)  # x-scale ("frequency")
    b_s = (2 * np.pi) / (temporal_resolution * 2 * (1 - winter_fraction))  # x-scale ("frequency")
    a_w = winter_balance * b_w / 2  # # y-scale ("amplitude")
    a_s = abs(summer_balance) * b_s / 2  # # y-scale ("amplitude")
    c = 0  # x-shift ("phase shift")
    d = 0  # y-shift ("vertical shift")

    # Calculate time intervals
    time_steps_annual = np.arange(0.5, temporal_resolution, 1).tolist()  # for full year
    winter_length = int(round(temporal_resolution * winter_fraction, 0))
    summer_length = int(temporal_resolution - winter_length)
    time_steps_winter = np.arange(0.5, 2 * winter_length, 1).tolist()
    time_steps_summer = np.arange(0.5, 2 * summer_length, 1).tolist()

    # calculate balance for each time interval in winter and then in summer periods
    balances = []
    for time_step in time_steps_winter[:winter_length]:
        balance = a_w * np.sin(b_w * time_step + c) + d
        balances.append(balance)
    for time_step in time_steps_summer[-1 * summer_length:]:
        balance = a_s * np.sin(b_s * time_step + c) + d
        balances.append(balance)

    # add to output dataframe
    interpolated_balances_df = pd.DataFrame({'TIME_STEP': time_steps_annual, 'BALANCE': balances})

    return interpolated_balances_df


def interpolate_daily_balances(
    bwsa_df: pd.DataFrame,
    alpha: float,
    winter_fraction: float = 0.5
) -> pd.DataFrame:
    """Function to interpolate daily balances using sine functions. Returns daily balances.

    Parameters:
        bwsa_df: DataFrame with Ba, Bw, Bs for selected glacier and time period.
        alpha: Climatic balance amplitude calculated from winter and summer balances.
        winter_fraction: Annual fraction of winter season, e.g. 8/12. By default, the fraction is set to equal
        lenght of winter and summer seasons.

    Returns:
        bd_df:Dataframe with columns DATE (date) and BALANCE (float).
    """
    # create empty dataframe to store results
    bd_df = pd.DataFrame(columns=['BALANCE'])
    bd_df.index.name = 'DATE'

    for index, row in bwsa_df.iterrows():
        # create time series for hydrological year
        row_start_date = f'{(int(row["Year"]) - 1)}-10-01'
        row_end_date = f'{int(row["Year"])}-09-30'
        row_dates = pd.Series(pd.date_range(row_start_date, row_end_date, freq='D'))
        row_numofdays = row_dates.index[-1] + 1

        # check if current year does have seasonal balances
        if np.isnan(row['WINTER_BALANCE']) or np.isnan(row['SUMMER_BALANCE']):
            # interpolate balances for current year climatic mass-balance amplitude and annual balances
            row_balances = sine_interpolation_from_mean_balances(
                annual_balance=row['ANNUAL_BALANCE'],
                balance_amplitude=alpha,
                winter_fraction=winter_fraction,
                temporal_resolution=row_numofdays
            )
        else:
            # interpolate balances for current year from winter and summer balances
            row_balances = sine_interpolation_from_seasonal_balances(
                winter_balance=row['WINTER_BALANCE'],
                summer_balance=row['SUMMER_BALANCE'],
                winter_fraction=winter_fraction,
                temporal_resolution=row_numofdays
            )
        # add annual to dataframe
        annual_df = pd.DataFrame({'BALANCE': row_balances['BALANCE'].tolist()}, index=row_dates)
        # append to series
        bd_df = pd.concat([bd_df, annual_df])

    return bd_df


def calc_mass_balance_amplitude(bwsa_df: pd.DataFrame) -> float:
    """Function to calculate glacier mass-balance amplitude based on available winter and summer balances.
        The term and calculation of glacier mass-balance amplitude are based on Cogley et al. (2011, UNESCO & IACS)
        and Braithwaite & Hughes (2020, Frontiers Earth Sciences).
        Returns mass-balance amplitude.

    Parameters:
       bwsa_df: DataFrame with winter, summer, and annual balances for selected glacier and time period.

    Returns:
       alpha: Climatic mass-balance amplitude calculated from Bw and abs(Bs).
    """
    alphas = ((bwsa_df['WINTER_BALANCE'] - bwsa_df['SUMMER_BALANCE']) / 2).abs()
    return alphas.mean()
