"""
Interpolate glacier mass balance from seasonal observations.
"""
import pandas as pd
import numpy as np


def sine_interpolation_from_mean_balances(balance_amplitude, annual_balance, temporal_resolution, winter_fraction):
    """Sine function to interpolate mass balance over the year.
        The function allows to set different lengths of winter/summer seasons.
        Returns balances at sub-seasonal rates.

    Parameters:
        balance_amplitude (float):Mass-balance amplitude, calculated from mean winter and mean abs summer balances.
        annual_balance (float):Mass balance, annual or mean annual value.
        temporal_resolution (float):Temporal resolution as desired for output, e.g. 12 (monthly) or 364 (daily).
        winter_fraction (float): annual fraction of winter season, e.g. 8/12.

    Returns:
        sub_annual_balances_df(float):Dataframe with columns TIME_STEP (float) and BALANCE (float).
    """
    # Define sine function parameters following the basic equation: y = f(x) = A sin(Bx + C) + D
    # Note that we need different sine functions for winter and summer, that can have different season lengths
    b_w = (2 * np.pi) / (temporal_resolution * 2 * winter_fraction)  # x-scale ("frequency")
    b_s = (2 * np.pi) / (temporal_resolution * 2 * (1 - winter_fraction))  # x-scale ("frequency")
    a_w = balance_amplitude * b_w / 2  # # y-scale ("amplitude")
    a_s = balance_amplitude * b_s / 2  # # y-scale ("amplitude")
    c = 0  # x-shift ("phase shift")
    d_w = annual_balance / temporal_resolution  # y-shift ("vertical shift")
    d_s = annual_balance / temporal_resolution  # y-shift ("vertical shift")

    # Calculate time intervals (note that winter + summer might not sum to 12 months due to different x-scales)
    time_steps_annual = np.arange(0.5, temporal_resolution, 1).tolist()  # for full year
    winter_length = int(round(temporal_resolution * winter_fraction, 0))
    summer_length = int(temporal_resolution - winter_length)
    time_steps_winter = np.arange(0.5, 2 * winter_length, 1).tolist()
    time_steps_summer = np.arange(0.5, 2 * summer_length, 1).tolist()

    # calculate balance for each time interval
    balances = []
    for time_step in time_steps_winter[:winter_length]:
        balance = a_w * np.sin(b_w * time_step + c) + d_w
        balances.append(balance)
    for time_step in time_steps_summer[-1 * summer_length:]:
        balance = a_s * np.sin(b_s * time_step + c) + d_s
        balances.append(balance)

    # add to output dataframe
    interpolated_balances_df = pd.DataFrame({'TIME_STEP': time_steps_annual, 'BALANCE': balances})

    return interpolated_balances_df


def sine_interpolation_from_seasonal_balances(winter_balance, summer_balance, temporal_resolution, winter_fraction):
    """Sine function to interpolate mass balance over the year.
       The function allows to set different lengths of winter/summer seasons.
       Returns balances at sub-seasonal rates.

    Parameters:
        winter_balance (float):Winter balance, for a given year or mean over a given period.
        summer_balance (float):Summer balance, for a given year or mean over a given period.
        temporal_resolution (float):Temporal resolution as desired for output, e.g. 12 (monthly) or 364 (daily).
        winter_fraction (float): annual fraction of winter season, e.g. 8/12.

    Returns:
        sub_annual_balances_df(float):Dataframe with columns TIME_STEP (float) and BALANCE (float).
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


def interpolate_daily_balances(bwsa_df, alpha, winter_fraction=6/12):
    """Function to interpolate daily balances using sine functions. Returns daily balances.

    Parameters:
        bwsa_df (object): DataFrame with Ba, Bw, Bs for selected glacier and time period.
        alpha (float): Climatic balance amplitude calculated from winter and summer balances.
        winter_fraction (float): Annual fraction of winter season, e.g. 8/12. By default, the fraction is set to equal
        lenght of winter and summer seasons.

    Returns:
        bd_df(float):Dataframe with columns DATE (date) and BALANCE (float).
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
            row_balances = sine_interpolation_from_mean_balances(alpha, row['ANNUAL_BALANCE'], row_numofdays,
                                                                    winter_fraction)
        else:
            # interpolate balances for current year from winter and summer balances
            row_balances = sine_interpolation_from_seasonal_balances(row['WINTER_BALANCE'], row['SUMMER_BALANCE'],
                                                                        row_numofdays, winter_fraction)
        # add annual to dataframe
        annual_df = pd.DataFrame({'BALANCE': row_balances['BALANCE'].tolist()}, index=row_dates)
        # append to series
        bd_df = pd.concat([bd_df, annual_df])

    return bd_df


def calc_mass_balance_amplitude(bwsa_df):
    """Function to calculate glacier mass-balance amplitude based on available winter and summer balances.
        The term and calculation of glacier mass-balance amplitude are based on Cogley et al. (2011, UNESCO & IACS)
        and Braithwaite & Hughes (2020, Frontiers Earth Sciences).
        Returns mass-balance amplitude.

    Parameters:
       bwsa_df (object): DataFrame with winter, summer, and annual balances for selected glacier and time period.

    Returns:
       alpha (float): Climatic mass-balance amplitude calculated from Bw and abs(Bs).
    """

    # calculate glacier mass-balance amplitude (using absolute values in case data provided with wrong signs)
    alpha = (abs(bwsa_df['WINTER_BALANCE'].mean()) + abs(bwsa_df['SUMMER_BALANCE'].mean())) / 2

    return alpha
