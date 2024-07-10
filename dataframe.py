import uuid
from typing import List, Union

import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from pandas import DataFrame as pandasDF
from polars import DataFrame as polarsDF
from sklearn.model_selection import train_test_split

from helpers import percentage as percentage_fn, timeit, memoryit

from skpm.config import EventLogConfig as elc
import skpm.event_logs as el


def preprocess_bpi(df: pl.DataFrame) -> pd.DataFrame:
    df = df.to_dummies(columns=[elc.activity])
    df = df.drop(elc.timestamp)
    return df


def get_all_bpi(engine="polars"):
    bpi_list = [el.BPI12, el.BPI13ClosedProblems, el.BPI13Incidents, el.BPI17, el.BPI19]
    dataframes = []
    for bpi in bpi_list:
        event = bpi()
        df = pl.from_pandas(event.log)
        df = preprocess_bpi(df)
        df = df if engine == "polars" else df.to_pandas()
        dataframes.append(df)
    return dataframes


def get_bpi12(engine="polars"):
    bpi12 = el.BPI12()
    df = pl.from_pandas(bpi12.log)
    df = preprocess_bpi(df)
    return df if engine == "polars" else df.to_pandas() if engine == "pandas" else None


def get_df(engine="polars"):
    df = (pl.read_ndjson("logs/ts-events.json")
          .rename({"ts": elc.timestamp, "id": elc.case_id, "event": elc.activity})
          .with_columns(pl.from_epoch(elc.timestamp, time_unit="s"))
          .sort(by=[elc.case_id, elc.timestamp]))
    df = df.to_dummies(columns=[elc.activity])
    df = df.drop(elc.timestamp)
    return df if engine == "polars" else df.to_pandas() if engine == "pandas" else None


def get_df_by_trace_length(engine="polars", df: Union[pl.DataFrame, pd.DataFrame] = None):
    if df is None:
        df: pd.DataFrame = get_df("pandas")
    elif isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    lengths = df.groupby(elc.case_id).size().reset_index().rename(columns={0: "length"})
    drop = lengths.length.value_counts() == 1  # drop the traces with count 1
    drop = drop[drop]
    lengths = lengths[~lengths.length.isin(drop.index)]
    return pl.from_pandas(lengths) if engine == "polars" else lengths if engine == "pandas" else None


def get_stratified_df_by_trace_length(engine="polars",
                                      df: Union[pl.DataFrame, pd.DataFrame] = None, random_state=42, size=1,
                                      stratify=True):
    lengths = get_df_by_trace_length(engine="pandas", df=df)
    # with stratification, it preserves the distribution of each class for training and testing sets according
    # to the main dataset
    _, stratified_lengths = train_test_split(lengths, test_size=size, random_state=random_state,
                                             stratify=lengths['length'] if stratify else None)
    return pl.from_pandas(
        stratified_lengths) if engine == "polars" else stratified_lengths if engine == "pandas" else None


def get_stratified_df_by_trace_length_split_by_percent(engine="polars", percents: List[float] = None,
                                                       df: Union[pl.DataFrame, pd.DataFrame] = None):
    lengths = get_df_by_trace_length(engine=engine, df=df)
    if percents is None:
        return lengths
    else:
        frames = []
        for dataset_size in percents:
            # with stratification, it preserves the distribution of each class for training and testing sets according
            # to the main dataset
            _, benchmark_split = train_test_split(lengths, test_size=dataset_size, random_state=42,
                                                  stratify=lengths.length)
            benchmark = df[df[elc.case_id].isin(benchmark_split[elc.case_id])]
            frames.append(benchmark)
        frames.append(lengths)  # adds the 100%
        return map(pl.from_pandas, frames) if engine == "polars" else frames if engine == "pandas" else None


def get_df_percentage(dataframe: pandasDF | polarsDF, percentage: int = 100):
    return dataframe[:percentage_fn(percentage, len(dataframe))]


def get_df_percentage_by_sklearn(dataframe: pandasDF | polarsDF, percentage: float = 1):
    if percentage == 1:
        return dataframe

    _, benchmark_split = train_test_split(dataframe, test_size=percentage, random_state=42,
                                          stratify=dataframe["length"])
    if isinstance(dataframe, pandasDF):
        return dataframe[dataframe[elc.case_id].isin(benchmark_split[elc.case_id])]
    if isinstance(dataframe, polarsDF):
        return dataframe.filter(pl.col(elc.case_id).is_in(benchmark_split[elc.case_id]))


def get_time_and_memory(df, agg_func, win_agg_func, percentages, percentage_func=get_df_percentage):
    """
    Get aggregation and window aggregation timings for a dataframe.

    Parameters:
    df (DataFrame): The dataframe to benchmark.
    agg_func (function): The aggregation function.
    win_agg_func (function): The window aggregation function.
    percentages (list): List of percentages to test.

    Returns:
    dict: Dictionary with percentage as keys and tuple of timings as values.
    """
    df_timings = {}
    df_memories = {}
    for percentage in percentages:
        subset_df = percentage_func(df, percentage)
        (_, agg_memory), agg_time = timeit(lambda: memoryit(lambda: agg_func(subset_df)))
        (_, win_agg_memory), win_agg_time = timeit(lambda: memoryit(lambda: win_agg_func(subset_df)))
        df_timings[percentage] = (agg_time, win_agg_time)
        df_memories[percentage] = (agg_memory, win_agg_memory)
    return df_timings, df_memories


def get_time_and_memory_by_trace_length(df, agg_func, win_agg_func, lengths: List[int]):
    """
    Get aggregation and window aggregation timings for a dataframe that contains trace length.

    Parameters:
    df (DataFrame): The dataframe to benchmark.
    agg_func (function): The aggregation function.
    win_agg_func (function): The window aggregation function.

    Returns:
    dict: Dictionary with percentage as keys and tuple of timings as values.
    """
    df_timings = {}
    df_memories = {}
    for trace_length in lengths:
        if isinstance(df, pl.DataFrame):
            subset_df = df.filter(pl.col('length') <= trace_length)
        else:
            subset_df = df[df['length'] <= trace_length]

        (_, agg_memory), agg_time = timeit(lambda: memoryit(lambda: agg_func(subset_df)))
        (_, win_agg_memory), win_agg_time = timeit(lambda: memoryit(lambda: win_agg_func(subset_df)))
        df_timings[trace_length] = (agg_time, win_agg_time)
        df_memories[trace_length] = (agg_memory, win_agg_memory)
    return df_timings, df_memories


def plot_timings(percentages, total, polars_agg_times, pandas_agg_times, polars_win_agg_times, pandas_win_agg_times,
                 x_label: str = None, y_label: str = None, title_1: str = None,
                 title_2: str = None):
    """
    Plot the aggregation and window aggregation timings.

    Parameters:
    percentages (list): List of percentages to test.
    polars_agg_times (list): List of polars aggregation times.
    pandas_agg_times (list): List of pandas aggregation times.
    polars_win_agg_times (list): List of polars window aggregation times.
    pandas_win_agg_times (list): List of pandas window aggregation times.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Aggregation results
    ax1.plot(percentages, polars_agg_times, label='Polars Agg', marker='o')
    ax1.plot(percentages, pandas_agg_times, label='Pandas Agg', marker='o')
    ax1.set_xlabel(f'Percentage of DataFrame (total {total})' if x_label is None else x_label)
    ax1.set_ylabel('Time (seconds)' if y_label is None else y_label)
    ax1.set_title('Aggregation Time Comparison (Polars vs Pandas)' if title_1 is None else title_1)
    ax1.legend()

    # Plot Window Aggregation results
    ax2.plot(percentages, polars_win_agg_times, label='Polars Win Agg', marker='o')
    ax2.plot(percentages, pandas_win_agg_times, label='Pandas Win Agg', marker='o')
    ax2.set_xlabel(f'Percentage of DataFrame (total {total})' if x_label is None else x_label)
    ax2.set_ylabel('Time (seconds)' if y_label is None else y_label)
    ax2.set_title('Window Aggregation Time Comparison (Polars vs Pandas)' if title_2 is None else title_2)
    ax2.legend()

    # Show plots
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{str(uuid.uuid4())[:8]}.png')


def plot_timings_bar(groups, values, labels=None):
    if labels is None:
        labels = [f'Round {i}' for i in range(len(values))]
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    ax.autoscale(enable=True)

    x = np.arange(len(groups))  # the label locations
    num_values = len(values)

    for i in range(num_values):
        ax.bar(x + i*width - width*(num_values-1)/2, values[i], width, label=labels[i])


# Adding labels, title, and legend
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Time')
    ax.set_title('Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()

    plt.show()
    fig.savefig(f'{str(uuid.uuid4())[:8]}.png')


def plot_memories(percentages, total, polars_agg_memory, pandas_agg_memory, polars_win_agg_memory,
                  pandas_win_agg_memory, x_label: str = None, y_label: str = None, title_1: str = None,
                  title_2: str = None):
    """
    Plot the aggregation and window aggregation timings.

    Parameters:
    percentages (list): List of percentages to test.
    polars_agg_memory (list): List of polars aggregation times.
    pandas_agg_memory (list): List of pandas aggregation times.
    polars_win_agg_memory (list): List of polars window aggregation times.
    pandas_win_agg_memory (list): List of pandas window aggregation times.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Aggregation results
    ax1.plot(percentages, polars_agg_memory, label='Polars Agg', marker='o')
    ax1.plot(percentages, pandas_agg_memory, label='Pandas Agg', marker='o')
    ax1.set_xlabel(f'Percentage of DataFrame (total {total})' if x_label is None else x_label)
    ax1.set_ylabel('Memory (MB)' if y_label is None else y_label)
    ax1.set_title('Aggregation Memory Comparison (Polars vs Pandas)' if title_1 is None else title_1)
    ax1.legend()

    # Plot Window Aggregation results
    ax2.plot(percentages, polars_win_agg_memory, label='Polars Win Agg', marker='o')
    ax2.plot(percentages, pandas_win_agg_memory, label='Pandas Win Agg', marker='o')
    ax2.set_xlabel(f'Percentage of DataFrame (total {total})' if x_label is None else x_label)
    ax2.set_ylabel('Memory (MB)' if y_label is None else y_label)
    ax2.set_title('Window Aggregation Memory Comparison (Polars vs Pandas)' if title_2 is None else title_2)
    ax2.legend()

    # Show plots
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{str(uuid.uuid4())[:8]}.png')
