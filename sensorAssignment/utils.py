import os
import pickle
from typing import Any, List, Tuple, Union
import itertools
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import ccf, adfuller, acf, pacf
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from statsmodels.graphics.tsaplots import plot_acf, acf
import networkx as nx

import matplotlib.lines as mlines

# original input
PATH_EXCEL_INPUT = 'data/2020_Sensor_Location_Assignment.xlsx'
PATH_PARQUET_INPUT = 'data/data.parquet'

# self created data that must be used again as input
PATH_INTERPOLATED_DATA = 'data/interpolated_sensors'
PATH_HIGH_CORR_PAIRS = 'data/high_correlating_pairs_numerical'
PATH_ALL_CORR_MATRICES = 'data/all_corr_matrices'

# output of results (plots / excels etc.)
# TODO: save plots
PATH_MAX_CORRS_DF = 'data/max_corrs_df.xlsx'
PATH_MIN_CORRS_DF = 'data/min_corrs_df.xlsx'
PATH_COMBINE_CORRS_DF = 'data/combine_corrs_df.xlsx'

PATH_SMOOTH_DF = 'data/smooth_df.xlsx'

PATH_RESPONSETIMES_DF = 'data/responsetimes_df.xlsx'
PATH_NETWORK_DF = 'data/network_df.xlsx'

DIR_VISUALISATIONS = 'figures/q4/'


## FILE STUFF
def _transform_and_save_to_parquet(frame: pd.DataFrame) -> None:
    """ Helper function.
        Saves dataframe to parquet file type, to load it next time but faster. """
    frame['UoM'] = frame['UoM'].astype('str')
    frame.to_parquet(PATH_PARQUET_INPUT, engine='pyarrow')


def load_raw_data() -> pd.DataFrame:
    if os.path.isfile(PATH_PARQUET_INPUT):
        print("Directly load parquet file fast. ")
        df = pd.read_parquet(PATH_PARQUET_INPUT)
    else:
        print("Load excel fast slowly and create parquet file for next time. ")
        df = pd.read_excel(PATH_EXCEL_INPUT)
        _transform_and_save_to_parquet(df)
    return df


def _load_pickle(path: str) -> Any:
    infile = open(path, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data


def load_interpolated_data():
    return _load_pickle(PATH_INTERPOLATED_DATA)


def load_high_corr_pairs():
    return _load_pickle(PATH_HIGH_CORR_PAIRS)


def load_all_corr_matrices():
    return _load_pickle(PATH_ALL_CORR_MATRICES)


def load_max_output():
    return _load_pickle(PATH_MAX_CORRS_DF)


def load_min_output():
    return _load_pickle(PATH_MIN_CORRS_DF)


def load_combine_output():
    return _load_pickle(PATH_COMBINE_CORRS_DF)


def load_smooth_df():
    return _load_pickle(PATH_SMOOTH_DF)


def load_responsetimes_df():
    return _load_pickle(PATH_RESPONSETIMES_DF)


def _save_pickle(var: Any, path: str) -> None:
    outfile = open(path, 'wb')
    pickle.dump(var, outfile)
    outfile.close()


def save_interpolated_data(var: Any) -> None:
    return _save_pickle(var, PATH_INTERPOLATED_DATA)


def save_high_corr_pairs(high_corr_pairs):
    return _save_pickle(high_corr_pairs, PATH_HIGH_CORR_PAIRS)


def save_all_corr_matrices(nested_matrix_list: List[np.ndarray]):
    return _save_pickle(nested_matrix_list, PATH_ALL_CORR_MATRICES)


def save_max_output(var: pd.DataFrame) -> None:
    return _save_pickle(var, PATH_MAX_CORRS_DF)


def save_combine_output(var: pd.DataFrame) -> None:
    return _save_pickle(var, PATH_COMBINE_CORRS_DF)


def save_min_output(var: pd.DataFrame) -> None:
    return _save_pickle(var, PATH_MIN_CORRS_DF)


def save_smooth_df(var: pd.DataFrame) -> None:
    return _save_pickle(var, PATH_SMOOTH_DF)


def save_responsetime_df(var: pd.DataFrame) -> None:
    return _save_pickle(var, PATH_RESPONSETIMES_DF)


def save_network_dfs(var: pd.DataFrame) -> None:
    return _save_pickle(var, PATH_NETWORK_DF)


## REGULAR FUNCTIONS
def flatten(nested_list: list):
    return [item for sublist in nested_list for item in sublist]


def get_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """ Get some statistics per tag of the dataset, like count, timestamp differences etc.  """
    res_df = pd.DataFrame({'Count': frame[['Tag', 'UoM']].value_counts()}).reset_index()
    res = []
    for tag in res_df['Tag'].unique().tolist():
        res.append(frame[frame['Tag'] == tag]['TS'].diff().value_counts())

    res_df['Diffs'] = res
    res_df['Nr_Diffs'] = res_df['Diffs'].apply(len)

    return res_df


def interpolate_sensor(frame: pd.DataFrame, tag_id: str, freq: str) -> pd.DataFrame:
    """ Takes the full dataset and a tag_id. Interpolates the whole 48-hours, on a given frequency.
        Default is linear interpolation.  """
    first_df = frame[frame['Tag'] == tag_id].set_index('TS')

    indices = pd.date_range(start='2020-06-12', end='2020-06-14', freq=freq)
    res_df = pd.DataFrame(index=indices)
    res_df = res_df.merge(first_df['avg'],
                          how='left',
                          left_index=True,
                          right_index=True
                          )
    res_df = res_df.interpolate(method='linear')
    return res_df


def moving_average(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """ Takes a dataframe and adds a column that smoothes the 'avg' column using moving average
    window (integer): indicates the moving average window"""
    df_moving_average = frame.copy()
    df_moving_average.reset_index(inplace=True)
    df_moving_average['ma_slope'] = df_moving_average['slope'].rolling(window, min_periods=1).mean()

    return df_moving_average


def get_derivative(frame: pd.DataFrame, col_name: str = 'slope') -> pd.DataFrame:
    """ Takes dataframe with one column holding sensor values and a timeseries index.
        Adds a column with the slope to the dataframe. """

    frame[col_name] = np.gradient(frame.values.reshape(len(frame)))

    return frame


def merge_frames(*args: pd.DataFrame, freq: str, drop_missing: bool = True, col_to_pick: str = 'avg') -> pd.DataFrame:
    """ Takes X numbers of dataframes and makes a new dataframe with a column for each sensor ('avg'). """
    indices = pd.date_range(start='2020-06-12', end='2020-06-14', freq=freq)
    res_df = pd.DataFrame(index=indices)  # create dataframe with the relevant time index

    sensor_id = 0
    for frame in args:  # loop over all dataframes are create separate column for 'avg' values for each sensor
        frame = frame.set_index('TS')
        res_df[f'sensor_{sensor_id}'] = frame[col_to_pick]
        sensor_id += 1

    if drop_missing:  # drop missing values in the data that came there because of interpolation
        res_df = res_df[~res_df.isnull().any(axis=1)]

    return res_df


def calc_corr_both_dirs(x, y, lag: int) -> np.array:
    """ x and y are 'array-like' """
    backwards = ccf(y, x)[:lag][::-1]
    forwards = ccf(x, y)[:lag]
    ccf_output = np.r_[backwards[:-1], forwards]
    return ccf_output


def plot_corr(correlation_array: np.array, lag: int, figsize: tuple = None, ci: bool = True) -> None:
    """ Plot the correlation function output.
        Plots both forward as backward. """
    f, ax = plt.subplots(figsize=figsize)
    ax.stem(np.arange(-lag + 1, lag), correlation_array, '-.')
    ax.set_xticks(np.arange(-lag + 1, lag))
    ax.locator_params(nbins=8)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation")

    if ci:
        sl = 2 / np.sqrt(lag)
        ax.plot([i for i in range(-lag + 1, lag)], list(np.ones(lag * 2 - 1) * sl), color='r')
        ax.plot([i for i in range(-lag + 1, lag)], list(np.ones(lag * 2 - 1) * -sl), color='r')

    plt.show()


def corr_two(input_sensors: List[str], interpolated_df: pd.DataFrame,
             max_lags: int, freq: str, data_col: str, plot: bool = True, output_arr: bool = False) -> pd.DataFrame:
    """ Full function that takes a list holding two sensors IDs and calculates the correlations. """
    res = []

    for sensor in input_sensors:
        interpolated_frame = interpolated_df[interpolated_df['Tag'] == sensor]
        res.append(interpolated_frame)

    merged_df = merge_frames(res[0], res[1], freq=freq, col_to_pick=data_col)

    ccf_output = calc_corr_both_dirs(x=merged_df.iloc[:, 0],
                                     y=merged_df.iloc[:, 1],
                                     lag=max_lags,
                                     )
    if plot:
        plot_corr(ccf_output, max_lags, figsize=(15, 10))

    if output_arr:
        return ccf_output
    else:
        # also return the lag of the correlation value
        # leng(list) /2 ) + 1 --> non lagged correlation. (this is the middle number of the list)
        # len(list) --> 2 x lag (+1)
        # create dataframe column with numbers [-lag, lag]
        return pd.DataFrame(list(zip(range(- max_lags + 1, max_lags), ccf_output)), columns=['Lag', 'Corr'])


def get_derivative_per_sensortype(sensor_type, interpolated_df, column_names, reset_index=True):
    """
    DESCRIPTION OF FUNCTION
    Loops over all sensors of a given sensor type. For each sensor calculates the derivative at every timestamp
    by calling the 'get_derivative' (utils) function. Appends the right columns back to those results. Finally,
    saves the derived dataframe for this sensor to the dataframe for all sensors of this sensor type.
    Return: dataframe containing all sensors of a specified type and for all these sensors the original data plus the
    derivative column.


    sensor_type (string): type of sensor e.g. 'flow'
    interpolated_df (pandas dataframe): dataframe that contains data to which moving average is applied to go from 5S sparse data to a full dataframe with average values of the defined moving average window.
    columns (list): list of all column names that need to be included in the dataframe that is created as the first command in this function.
    reset_index (boolean): False indicates that index of dataframe is not reset and thus TS (timestamp) is used as index column. True means that a new index column is placed and TS is saved as column in the dataframe.

    """

    der_df = pd.DataFrame(columns=column_names)
    # create a new dataframe to which ma dataframes can be appended

    sensors_of_type = list(interpolated_df['Tag'][interpolated_df['UoM'] == sensor_type].unique())

    for sensor in sensors_of_type:

        # isolate rows of a specific sensor
        sensor_data = interpolated_df[['avg', 'TS']][interpolated_df['Tag'] == sensor]

        # set index on TimeStamp column
        sensor_data = sensor_data.set_index('TS')

        df_slope = get_derivative(sensor_data)

        # append rows to slope dataframe
        df_slope['Tag'] = sensor
        df_slope['UoM'] = sensor_type  # UoM is same in all rows for 1 Tag. So set this value in the UoM column

        if reset_index:
            df_slope.reset_index(inplace=True)
            df_slope = df_slope.rename(columns={'index': 'TS'})
        else:
            # drop the column TS which is the index column in this scenario
            der_df = der_df.drop(['TS'], axis=1)

        der_df = der_df.append(df_slope)

    return der_df


def smooth_sensors_of_type(sensor_type, derivative_df, column_names, window):
    """
    DESCRIPTION OF FUNCTION
     Loops over all sensors of a given sensor type. Applies Moving Average with a given MA rolling window to smooth
     the derivative (slope) values from the derivative_df for a specific type of sensor.
     Return: dataframe with column appended with smoothed derivative values.

    :sensor_type (string): type of sensor e.g. 'flow'
    :derivative_df (pandas dataframe): dataframe that contains the derived data that can be smoothed by applying moving average.
    :columns (list): list of all column names that need to be included in the dataframe that is created as the first command in this function.
    :reset_index (boolean): False indicates that index of dataframe is not reset and thus TS (timestamp) is used as index column.
                        True means that a new index column is placed and TS is saved as column in the dataframe.
    """

    ma_df = pd.DataFrame(columns=column_names)
    # create a new dataframe to which ma dataframes can be appended

    sensors_of_type = list(derivative_df['Tag'][derivative_df['UoM'] == sensor_type].unique())

    for sensor in sensors_of_type:
        # isolate rows of a specific sensor
        sensor_data = derivative_df[['slope', 'TS', 'avg']][derivative_df['Tag'] == sensor]

        # set index on TimeStamp column
        sensor_data = sensor_data.set_index('TS')

        # if set to true, apply moving average first to smooth the data.
        sensor_data = moving_average(sensor_data, window)
        # sensor_data = ma.set_index('TS')

        # append rows to ma_data dataframe
        sensor_data['Tag'] = sensor
        sensor_data['UoM'] = sensor_type  # UoM is same in all rows for 1 Tag. So set this value in the UoM column

        ma_df = ma_df.append(sensor_data)

    return ma_df


def filterout_top_rows(pct: float, sensor_df: pd.DataFrame):
    """
    To remove the delta peaks from the derived signals.
    The % of rows to be removed is a parameter in this function. This percentage determines how much of the top and bottom will be removed.
    E.g. 5% means top highest 5% AND top lowest (likely negative) 5% are removed.

    # PARAMETERS
    pct: float indicating the percentage of top rows that is to be filtered out
    sensor_df: pandas dataframe of a specific sensor's derivative dataframe

    :returns
    original dataframe with rows filtered out that fall in the top x % highest and x % lowest (negative) values.
    """

    # top 5% largest values
    n = round(pct * len(sensor_df))
    largest = list(sensor_df['slope'].nlargest(n))
    smallest = list(sensor_df['slope'].nsmallest(n))
    # top_rows contains all slopes within top 5% highest & lowest.
    top_rows = largest + smallest

    # add column that replaces top rows' slope with NaN as value
    sensor_df['_slope'] = np.where(np.isin(sensor_df['slope'], top_rows), 'NaN', sensor_df['slope'])

    # filter out rows with NaN _slopes
    sensor_df = sensor_df.loc[(sensor_df['_slope'] != 'NaN') & (sensor_df['_slope'] != 'nan')]

    return sensor_df


def calc_sensor_autocorelation(sensor: str, smoothend_df: pd.DataFrame, MAX_LAGS_ONE_DIR: int, data_col: str):
    """
    # DESCRIPTION
    Takes a dataframe with derivatives of every timestamp for one specified sensor. Calculates the auto-correlation
    and squared auto-correlation at a specified number of time lags.
    # PARAMETERS
    MAX_LAGS_ONE_DIR: integer that specifies the number of lags for which auto-correlation is to be calculated
    data_col: column for which the time serie is to be isolated.

    # RETURNS
    A dataframe for a specific sensor with the auto-correlation and squared auto-correlation for x time lags.


    """

    # Get auto correlation values for one specified sensor
    tmp_df = get_one_ts(smoothend_df,
                        sensor=sensor, data_col=data_col
                        )
    acorrs, conf_int = pacf(x=tmp_df,
                            nlags=MAX_LAGS_ONE_DIR,
                            alpha=0.05)

    # add results to dictionary before converting dictionary to dataframe
    acorr_dict = {'auto_corr': acorrs, 'Lag': [i for i in range(MAX_LAGS_ONE_DIR + 1)],
                  'Tag': [sensor] * (MAX_LAGS_ONE_DIR + 1)}

    # convert to pandas dataframe
    acorr_df = pd.DataFrame(acorr_dict)

    # add squared auto correlation column
    acorr_df['squared_autocorr'] = acorr_df['auto_corr'] * acorr_df['auto_corr']

    # drop row with lag = 0 as auto correlation is there always 1.0
    acorr_df = acorr_df[1:].copy()

    return acorr_df


def get_response_time_sensor(autocorrelated_df: pd.DataFrame):
    """
    # DESCRIPTION
    Determines the response time of a specific sensor. The explanation behind the formula can be found in the report.

    # RETURNS
    A float value that indicates the response time.

    """
    # determine the total sum of squared autocorrelated values
    summed = autocorrelated_df['squared_autocorr'].sum()

    # calculate the distribution by dividing the squared autocorrelation at every lag by the total sum of autocorrelation
    autocorrelated_df['Likelihood'] = autocorrelated_df['squared_autocorr'] / summed

    # 
    autocorrelated_df['Likelihood_*_Lag'] = (autocorrelated_df['distribution'] * autocorrelated_df['Lag'])

    # determine the response time of the sensor
    response_time = (autocorrelated_df['Probability_*_Lag'].sum() * 5 * 2) + 5

    return response_time, autocorrelated_df


def calc_response_time_sensor(sensor: str, frame: pd.DataFrame, MAX_LAGS_ONE_DIR, percent: float):
    """ This function combines all steps required to calculate the response time of a single sensor.
    The calculated response time is then appended to the general response_times dataframe"""

    # filter initial dataframe on locking only rows of the sensor in question
    sensor_df = frame.loc[frame['Tag'] == sensor].copy()

    # filter out top rows
    filtered_sensor_df = filterout_top_rows(percent, sensor_df)

    autocorr_df = calc_sensor_autocorelation(sensor, filtered_sensor_df, MAX_LAGS_ONE_DIR, data_col='avg')

    # determine the response time of the sensor
    response_time = get_response_time_sensor(autocorr_df)

    return response_time, sensor


def get_all_sensor_pairs(series: pd.Series) -> List[list]:
    """
    # DESCRIPTION


    # PARAMETERS


    # RETURNS



    """
    all_sensors = series.unique().tolist()
    all_combinations = list(itertools.product(all_sensors,
                                              all_sensors))  # take the inner product of the list of all sensors to obtain all possible sensor pairs
    all_sensor_pairs = [list(i) for i in all_combinations if
                        i[0] != i[1]]  # drop all sensor pairs with duplicate tags (a12345, a12345)
    sensor_pairs = []

    for pair in all_sensor_pairs:
        if pair[::-1] not in sensor_pairs:  # drop all duplicate sensor pairs
            sensor_pairs.append(pair)

    return sensor_pairs


def get_correlating_sensors(sensor_pairs: list, corr_cutoff: float, frame: pd.DataFrame, freq: str) -> dict:
    """
    # DESCRIPTION

    # PARAMETERS


    # RETURNS



    """
    max_corrs = {}

    # pair must be a list of 2 sensor tag strings
    for pair in tqdm(sensor_pairs):
        # call correlation function that calculates the correlation at various lags for a given pair of sensors
        corrs = corr_two(input_sensors=pair,
                         interpolated_df=frame,
                         plot=False,
                         max_lags=int(600 / int(freq[:-1])),
                         # take 600 seconds (10 minutes), divide this by the number of seconds resampled to get nr lags
                         freq=freq,
                         )

        # obtain highest absolute correlation value over all lags
        max_corr = max(list(corrs['Corr']), key=abs)

        # correlation must be higher/lower than the given cut off
        if max_corr > corr_cutoff or max_corr < - corr_cutoff:
            # as there are like nine decimals we can find the lag based on the correlation value.
            # the max corr value with corresponding lag are appended to a dictionary with the sensor pair as key
            max_corrs[tuple(pair)] = tuple(corrs.loc[corrs['Corr'] == max_corr].values.tolist()[0])

    return max_corrs


def make_max_corrs_df(max_corrs: dict, freq: str) -> pd.DataFrame:
    """
    # DESCRIPTION

    # PARAMETERS


    # RETURNS



    """
    # Get positive integers from string
    period = [int(s) for s in freq if s.isdigit()]
    print(f"Period: {period}")
    maxs = pd.DataFrame.from_dict(max_corrs, orient='index', columns=['Lag', 'Max_Corr'])
    # calculate relative time delay from Lag.
    # Note, this time delay is based on the aggregated lags
    # --> so, actual time delay could be different but is very hard to pick up.
    maxs['TimeDelay(min)'] = maxs['Lag'] * (period[0])
    maxs.reset_index().rename(columns={'index': 'Pair'})
    return maxs


def plot_sensor_pair_ts_separate(sensor_pair: tuple, smoothend_df: pd.DataFrame) -> None:
    """ Takes a pair of sensors and a smoothened dataframe.
        Plots both the sensors as an individual time series, to be able to compare the structure with one another. """
    frame1 = smoothend_df[smoothend_df['Tag'] == sensor_pair[0]]
    frame1 = frame1.set_index('TS')
    frame1['avg'].plot()
    plt.show()

    frame2 = smoothend_df[smoothend_df['Tag'] == sensor_pair[1]]
    frame2 = frame2.set_index('TS')
    frame2['avg'].plot()
    plt.show()


def calc_corr_matrices(unique_sensor_list: list, smoothend_df: pd.DataFrame, max_lags_one_dir: int,
                       freq: str, data_col: str) -> Tuple[Any, Any, Any, Any, Any]:
    """
    # DESCRIPTION

    # PARAMETERS


    # RETURNS



    """
    print("Calculating full correlation matrix.")
    print("tqdm shows number of columns completed.")
    full_corr_matrix = np.array(
        [[corr_two([one, two], smoothend_df, max_lags=max_lags_one_dir, freq=freq, plot=False, output_arr=True,
                   data_col=data_col) for
          one in unique_sensor_list]
         for two in tqdm(unique_sensor_list)]
    )

    shape = full_corr_matrix.shape[:2]
    empty_matrix = np.zeros(shape=shape)

    max_matrix = empty_matrix.copy()
    max_lag_matrix = empty_matrix.copy()
    min_matrix = empty_matrix.copy()
    min_lag_matrix = empty_matrix.copy()

    print("Creating max and min matrices and corresponding lag matrices")
    print("tqdm shows number of columns completed.")
    for idx_nr in tqdm(range(full_corr_matrix.shape[0])):
        for col_nr in range(full_corr_matrix.shape[1]):
            max_matrix[idx_nr][col_nr] = max(full_corr_matrix[idx_nr][col_nr])
            max_lag_matrix[idx_nr][col_nr] = np.argmax(full_corr_matrix[idx_nr][col_nr])
            min_matrix[idx_nr][col_nr] = min(full_corr_matrix[idx_nr][col_nr])
            min_lag_matrix[idx_nr][col_nr] = np.argmin(full_corr_matrix[idx_nr][col_nr])

    return full_corr_matrix, max_matrix, max_lag_matrix, min_matrix, min_lag_matrix


def plot_matrix(matrix, title=None, figsize=(12, 12), cmap='Reds',
                xticklabels='auto', yticklabels='auto'):
    """
    # DESCRIPTION

    # PARAMETERS


    # RETURNS



    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(matrix,
                ax=ax,
                linewidth=0.5,
                cmap=cmap,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                )
    ax.set_title(title)
    plt.show()


def matrix_to_df(matrix: np.ndarray, sensor_list: list) -> pd.DataFrame:
    return pd.DataFrame(data=matrix,
                        index=sensor_list,
                        columns=sensor_list,
                        )


def get_one_ts(smoothend_df: pd.DataFrame, sensor: str, dropna=True, data_col='ma_slope') -> pd.DataFrame:
    """ Gets one time series from the smoothend dataframe.  """
    tmp_df = smoothend_df[smoothend_df['Tag'] == sensor]
    tmp_df = tmp_df.set_index('TS')
    tmp_df = tmp_df[data_col]
    if dropna:
        tmp_df = tmp_df.dropna()
    return tmp_df


def calc_p_values_stationarity(smoothend_df: pd.DataFrame, sensor_list: list, data_col: str = 'ma_slope',
                               plot=True, title: str = None) -> list:
    """ Executes Augmented Dickey Fuller test, to check for stationarity.
        Also plots the p values. """
    print(f"Using data_col: {data_col}")
    p_values = []
    for sensor in tqdm(sensor_list):
        tmp_df = get_one_ts(smoothend_df, sensor, data_col=data_col)
        tmp_df = tmp_df.dropna()
        results_adf = adfuller(tmp_df)
        p_values.append(results_adf[1])

    if plot:
        sns.scatterplot(x=[i for i in range(len(p_values))], y=p_values)
        plt.title(title)
        plt.xlabel("Sensor ID")
        plt.ylabel("p-value")

    return p_values


def prewhitening_smoothend_df(smoothend_df: pd.DataFrame, dropna=False, data_col='ma') -> pd.DataFrame:
    """
    # DESCRIPTION

    # PARAMETERS


    # RETURNS



    """
    sensor_list = smoothend_df['Tag'].unique().tolist()

    res = []
    for sensor in sensor_list:
        tmp_df = get_one_ts(smoothend_df=smoothend_df,
                            sensor=sensor,
                            dropna=dropna,
                            data_col=data_col,
                            )
        res.append(tmp_df.diff().tolist())

    res = flatten(res)
    smoothend_df['prewhitening'] = res

    return smoothend_df


def thres_on_matrix(matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return matrix[matrix > threshold]


def get_output(matrix: pd.DataFrame, lag_matrix: pd.DataFrame, sensor_list: list) -> pd.DataFrame:
    """
    # DESCRIPTION

    # PARAMETERS


    # RETURNS



    """
    matrix_used = matrix.copy()
    # replace diagonal values with nan value
    for i in range(len(matrix_used)):
        matrix_used.iloc[i, i] = np.nan

    pair_indices = list(zip(*np.where(matrix_used.notnull())))  # get indices of pairs that are not null
    pair_sensors = [(sensor_list[x], sensor_list[y]) for x, y in pair_indices]  # get sensor ids of corresponding pairs

    all_values = [matrix.loc[pair[0], pair[1]] for pair in pair_sensors]  # fetch the correlation values from the matrix
    all_lags = [lag_matrix.loc[pair[0], pair[1]] for pair in pair_sensors]  # fetch the lag values from the matrix

    # create dataframe with the output
    results_df = pd.DataFrame(index=pair_sensors, data={
        'correlation': all_values,
        'lag': all_lags
    })

    return results_df


def create_type_mapping(smoothend_df: pd.DataFrame) -> dict:
    """
    # DESCRIPTION

    # PARAMETERS


    # RETURNS



    """
    first_dict = smoothend_df[['Tag', 'UoM']].drop_duplicates().set_index('Tag').to_dict(orient='index')
    type_mapping = {}
    for key, value in first_dict.items():
        type_mapping[key] = first_dict[key]['UoM']
    return type_mapping


def get_hugely_interpolated_sensors() -> list:
    """
    # DESCRIPTION

    # PARAMETERS


    # RETURNS



    """
    df = load_raw_data()
    df_grouped = df.groupby('Tag').count()
    df_grouped['Percentage Data Points'] = df_grouped['TS'].apply(lambda x: x / (60 * 60 * 48 / 5))
    df_grouped['Percentage Missing'] = 1 - df_grouped['Percentage Data Points']
    return df_grouped[df_grouped['TS'] < 50].index.tolist()


def get_plt_colors() -> np.array:
    """ Create array with approx 150 different colors from matplotlib. """
    mycolors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    mycolors = list(mycolors.keys())  # only take color names
    for col in ['white', 'whitesmoke', 'w']:  # delete non-visible colors
        mycolors.pop(mycolors.index(col))
    mycolors = np.array(mycolors)  # create array for proper plotting
    return mycolors


def plot_autocorrs(frame, sensor_list, data_col, max_lags_one_dir, title_addition):
    """
    # DESCRIPTION

    # PARAMETERS


    # RETURNS



    """
    for sensor in sensor_list:
        tmp_df = get_one_ts(smoothend_df=frame,
                            sensor=sensor,
                            data_col=data_col,
                            )
        plot_acf(x=tmp_df,
                 lags=max_lags_one_dir,
                 alpha=0.05)
        plt.title(f"{sensor}: {title_addition}")
        plt.xlabel("Time Lag")
        plt.ylabel("Autocorrelation Value")
        plt.show()


def check_high_autocorr(sensor, smoothend_df):
    """ Checks the number of high autocorrelations for a specific sensor and assign labels accordingly. """
    single_ts = get_one_ts(smoothend_df, sensor)
    acf_value = acf(single_ts,
                    nlags=15,
                    fft=False)
    count = sum(acf_value > 0.5)

    if count <= 5:
        return 0
    elif 5 < count <= 10:
        return 0.5
    elif count > 10:
        return 1


def check_sign(lst: list) -> list:
    """ Checks whether the sign of the lag is positive, negative, or zero. """
    res = []
    for value in lst:
        if value < 0:
            res.append(-1)
        elif value > 0:
            res.append(1)
        elif value == 0:
            res.append(0)
        else:
            raise ValueError
    return res


def plot_pairs(sensor_pair: list, smoothend_df: pd.DataFrame, n_start=0, n_end=2000):
    """ Plot small part of pair of time series in one plot.  """
    fig, ax = plt.subplots()
    smoothend_df[smoothend_df['Tag'] == sensor_pair[0]].set_index('TS')['ma_slope'][n_start:n_end].plot(ax=ax)
    smoothend_df[smoothend_df['Tag'] == sensor_pair[1]].set_index('TS')['ma_slope'][n_start:n_end].plot(ax=ax)
    plt.legend(sensor_pair)
    plt.show()


def get_all_sensors_from_df(frame):
    all_sensors = []
    for pair in frame.index.tolist():
        if pair[0] not in all_sensors:
            all_sensors.append(pair[0])
        if pair[1] not in all_sensors:
            all_sensors.append(pair[1])
    return all_sensors


def create_network_dfs(output_df: pd.DataFrame) -> Tuple[pd.DataFrame]:
    """ Creates dataframes ready for network analysis. """
    network_df = output_df.copy()
    network_df = network_df.reset_index()
    network_df = network_df.rename(columns={
        'index': 'pairs'
    })
    new_df = pd.DataFrame(data=network_df['pairs'].tolist(),
                          columns=['node_one', 'node_two'],
                          )
    network_df = network_df.drop('pairs', axis=1)
    new_df = pd.concat([new_df, network_df], axis=1)

    edges_df = new_df[['node_one', 'node_two', 'correlation', 'lag', 'sign']]

    rename_dict = {
        'node_one': 'node',
        'node_two': 'node',
        'type_one': 'type',
        'type_two': 'type',
        'huge_interpolated_one': 'huge_interpolated',
        'huge_interpolated_two': 'huge_interpolated',
        'autocorr_one': 'autocorr',
        'autocorr_two': 'autocorr',
    }

    nodes_df_one = new_df[['node_one', 'type_one', 'huge_interpolated_one', 'autocorr_one']]
    nodes_df_one = nodes_df_one.rename(columns=rename_dict)

    nodes_df_two = new_df[['node_two', 'type_two', 'huge_interpolated_two', 'autocorr_two']]
    nodes_df_two = nodes_df_two.rename(columns=rename_dict)

    nodes_df = pd.concat([nodes_df_one, nodes_df_two])
    nodes_df = nodes_df.drop_duplicates()

    return new_df, edges_df, nodes_df


def create_color_on_attr(frame: pd.DataFrame, col: str) -> np.array:
    """ Creates a color array based on a node or edge attribute. """
    type_list = frame[col].to_list()
    unique_type_list = pd.Series(type_list).unique().tolist()

    color_mapping = {}
    for i, key in enumerate(unique_type_list):
        color_mapping[key] = i

    type_list_int = [color_mapping[typ] for typ in type_list]
    # fetch colors
    mycolors = get_plt_colors()
    type_color_array = mycolors[type_list_int]

    return type_color_array


def create_full_graph(edges_frame: pd.DataFrame, nodes_frame: pd.DataFrame, with_labels: bool = False, circular=False,
                      savefig=False):
    # create color based on type, add to node dataframe and create list of nodes.
    type_color_array = create_color_on_attr(nodes_frame, 'type')
    nodes_frame['type_color'] = type_color_array
    nodelist = nodes_frame['node'].tolist()

    # create graph
    G = nx.from_pandas_edgelist(df=edges_frame,
                                source='node_one',
                                target='node_two',
                                edge_attr=['correlation', 'lag', 'sign']
                                )

    # add node attributes
    node_dict = nodes_frame[['node', 'type', 'type_color']].set_index('node').to_dict(orient='index')
    nx.set_node_attributes(G, node_dict)

    # get edges and lag values
    edgelist, lag_values = zip(*nx.get_edge_attributes(G, 'lag').items())

    # get node legend information
    type_names = nodes_frame[['type', 'type_color']].drop_duplicates()['type'].tolist()
    type_colors = nodes_frame[['type', 'type_color']].drop_duplicates()['type_color'].tolist()
    patch_list = []
    for col, name in zip(type_colors, type_names):
        patch_list.append(mpatches.Patch(color=col, label=name))

    # make plot basics
    plt.figure(figsize=(20, 10))
    if circular:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # fill in plot
    if with_labels:
        labels = nx.draw_networkx(G, pos=pos, with_labels=True)
    nodes = nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=type_color_array)
    edges = nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color=lag_values, edge_cmap=plt.cm.coolwarm,
                                   width=2, edge_vmin=-2, edge_vmax=2)
    plt.colorbar(edges)
    plt.legend(patch_list, type_names)
    if savefig:
        plt.savefig(DIR_VISUALISATIONS + str(time.time()) + '.PNG')

    return G


def create_separated_graphs(graph, edges_frame: pd.DataFrame, nodes_frame: pd.DataFrame,
                            with_labels: bool = False, figsize=None, circular=False, savefig=False) -> list:
    separated_graphs = list(nx.connected_components(graph))
    separated_graphs = [list(graphlist) for graphlist in separated_graphs]

    separated_graphs_node_dfs = [nodes_frame[nodes_frame['node'].isin(separated_graphs[i])] for i in
                                 range(len(separated_graphs))]

    separated_graphs_edge_dfs = [edges_frame[edges_frame['node_one'].isin(separated_graphs[i])] for i in
                                 range(len(separated_graphs))]

    all_graphs = []
    for single_edges_df, single_nodes_df in zip(separated_graphs_edge_dfs, separated_graphs_node_dfs):
        # create graph from edges
        G_separated = nx.from_pandas_edgelist(df=single_edges_df,
                                              source='node_one',
                                              target='node_two',
                                              edge_attr=['correlation', 'lag', 'sign']
                                              )

        node_color = single_nodes_df['type_color'].to_numpy()
        nodelist = single_nodes_df['node'].to_numpy()

        # get edges and lag values
        edgelist, lag_values = zip(*nx.get_edge_attributes(G_separated, 'lag').items())

        # get node legend information
        type_names = nodes_frame[['type', 'type_color']].drop_duplicates()['type'].tolist()
        type_colors = nodes_frame[['type', 'type_color']].drop_duplicates()['type_color'].tolist()
        patch_list = []
        for col, name in zip(type_colors, type_names):
            patch_list.append(mpatches.Patch(color=col, label=name))

        # make plot basics
        plt.figure(figsize=figsize)
        if circular:
            pos = nx.circular_layout(G_separated)
        else:
            pos = nx.spring_layout(G_separated, seed=42)

        # fill in plot
        if with_labels:
            labels = nx.draw_networkx(G_separated, pos=pos, with_labels=True)

        vmin = -2
        vmax = 2

        edges_for_legend = nx.draw_networkx_edges(G_separated, pos, edgelist=edgelist, edge_color=lag_values,
                                                  edge_cmap=plt.cm.coolwarm,
                                                  edge_vmin=vmin, edge_vmax=vmax,
                                                  width=2, arrowsize=40)

        G_separated = G_separated.to_directed()

        nodes = nx.draw_networkx_nodes(G_separated, pos, nodelist=nodelist, node_color=node_color)
        edges = nx.draw_networkx_edges(G_separated, pos, edgelist=edgelist, edge_color=lag_values,
                                       edge_cmap=plt.cm.coolwarm,
                                       edge_vmin=vmin, edge_vmax=vmax,
                                       width=2, arrowsize=40)

        plt.colorbar(edges_for_legend)
        plt.legend(patch_list, type_names)
        if savefig:
            plt.savefig(DIR_VISUALISATIONS + str(time.time()) + '.PNG')

        all_graphs.append(G_separated)

    return all_graphs


def create_degree_df(graph):
    graph = graph.to_directed()
    return pd.DataFrame(
        index=graph.nodes(),
        data={
            'total': np.array(graph.degree)[:, 1],
            'in': np.array(graph.in_degree)[:, 1],
            'out': np.array(graph.out_degree)[:, 1],
        }
    )


def plot_stats_of_output_scores(output_df, savefig=False):
    output_df['correlation'].plot.hist()
    plt.title("Correlation frequency count")
    if savefig:
        plt.savefig(DIR_VISUALISATIONS + str(time.time()) + '.PNG')
    plt.show()
    output_df['lag'].plot.hist()
    plt.title("Lag count at which the maximum absolute correlation is found")
    if savefig:
        plt.savefig(DIR_VISUALISATIONS + str(time.time()) + '.PNG')
    plt.show()
    output_df['sign'].plot.hist()
    x_ticks = [-1, 0, 1]
    plt.xticks(x_ticks)
    plt.title("Sign of the lag count at which the maximum absolute correlation is found")
    if savefig:
        plt.savefig(DIR_VISUALISATIONS + str(time.time()) + '.PNG')
    plt.show()
