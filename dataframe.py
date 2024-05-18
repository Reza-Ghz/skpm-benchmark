import polars as pl
from matplotlib import pyplot as plt
from pandas import DataFrame as pandasDF
from polars import DataFrame as polarsDF
from helpers import percentage as percentage_fn, timeit

from skpm.config import EventLogConfig as elc


def get_df(engine="polars"):
    df = (pl.read_ndjson("data/ts-events.json")
          .rename({"ts": elc.timestamp, "id": elc.case_id, "event": elc.activity})
          .with_columns(pl.from_epoch(elc.timestamp, time_unit="s"))
          .sort(by=[elc.case_id, elc.timestamp]))
    df = df.to_dummies(columns=[elc.activity])
    df = df.drop(elc.timestamp)
    return df if engine == "polars" else df.to_pandas() if engine == "pandas" else None


def get_df_percentage(dataframe: pandasDF | polarsDF, percentage: int = 100):
    return dataframe[:percentage_fn(percentage, len(dataframe))]


def get_timings(df, agg_func, win_agg_func, percentages):
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
    for percentage in percentages:
        subset_df = get_df_percentage(df, percentage)
        agg_time = timeit(lambda: agg_func(subset_df))
        win_agg_time = timeit(lambda: win_agg_func(subset_df))
        df_timings[percentage] = (agg_time, win_agg_time)
    return df_timings


def plot_timings(percentages, total, polars_agg_times, pandas_agg_times, polars_win_agg_times, pandas_win_agg_times):
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
    ax1.set_xlabel(f'Percentage of DataFrame (total {total})')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Aggregation Time Comparison (Polars vs Pandas)')
    ax1.legend()

    # Plot Window Aggregation results
    ax2.plot(percentages, polars_win_agg_times, label='Polars Win Agg', marker='o')
    ax2.plot(percentages, pandas_win_agg_times, label='Pandas Win Agg', marker='o')
    ax2.set_xlabel(f'Percentage of DataFrame (total {total})')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Window Aggregation Time Comparison (Polars vs Pandas)')
    ax2.legend()

    # Show plots
    plt.tight_layout()
    plt.show()
