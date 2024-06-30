import pandas as pd
import polars as pl

from dataframe import get_df, get_time_and_memory, plot_timings, get_df_by_trace_length, get_df_percentage_by_sklearn, \
    plot_memories, get_all_bpi
from processing import agg_pandas, agg_polars, win_agg_polars, win_agg_pandas


def benchmark_by_cases(polars_df: pl.DataFrame, pandas_df: pd.DataFrame):
    total = len(polars_df)
    percentages = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    # percentage_labels = [f'{p}({percentage(p, len(polars_df))})' for p in percentages]

    # Get timings for Polars and Pandas
    polars_timings, polars_memory = get_time_and_memory(polars_df, agg_polars, win_agg_polars, percentages)
    pandas_timings, pandas_memory = get_time_and_memory(pandas_df, agg_pandas, win_agg_pandas, percentages)

    # Extract timings for plotting
    polars_agg_times = [polars_timings[p][0] for p in percentages]
    polars_win_agg_times = [polars_timings[p][1] for p in percentages]
    pandas_agg_times = [pandas_timings[p][0] for p in percentages]
    pandas_win_agg_times = [pandas_timings[p][1] for p in percentages]

    # Extract timings for plotting
    polars_agg_memory = [polars_memory[p][0] for p in percentages]
    polars_win_agg_memory = [polars_memory[p][1] for p in percentages]
    pandas_agg_memory = [pandas_memory[p][0] for p in percentages]
    pandas_win_agg_memory = [pandas_memory[p][1] for p in percentages]

    # Plot timings
    plot_timings(percentages, total, polars_agg_times, pandas_agg_times, polars_win_agg_times,
                 pandas_win_agg_times)
    # Plot Memory
    plot_memories(percentages, total, polars_agg_memory, pandas_agg_memory, polars_win_agg_memory,
                  pandas_win_agg_memory)


def benchmark_by_trace_lengths(polars_df: pl.DataFrame, pandas_df: pd.DataFrame):
    total = len(polars_df)
    percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                   0.95, 1.0]

    # Get timings for Polars and Pandas
    polars_timings, polars_memory = get_time_and_memory(polars_df, agg_polars, win_agg_polars, percentages,
                                                        get_df_percentage_by_sklearn)
    pandas_timings, pandas_memory = get_time_and_memory(pandas_df, agg_pandas, win_agg_pandas, percentages,
                                                        get_df_percentage_by_sklearn)

    # Extract timings for plotting
    polars_agg_times = [polars_timings[p][0] for p in percentages]
    polars_win_agg_times = [polars_timings[p][1] for p in percentages]
    pandas_agg_times = [pandas_timings[p][0] for p in percentages]
    pandas_win_agg_times = [pandas_timings[p][1] for p in percentages]

    # Extract timings for plotting
    polars_agg_memory = [polars_memory[p][0] for p in percentages]
    polars_win_agg_memory = [polars_memory[p][1] for p in percentages]
    pandas_agg_memory = [pandas_memory[p][0] for p in percentages]
    pandas_win_agg_memory = [pandas_memory[p][1] for p in percentages]

    # Plot timings
    plot_timings(percentages, total, polars_agg_times, pandas_agg_times, polars_win_agg_times,
                 pandas_win_agg_times)
    # Plot Memory
    plot_memories(percentages, total, polars_agg_memory, pandas_agg_memory, polars_win_agg_memory,
                  pandas_win_agg_memory)


def main():
    # polars_df = get_df("polars")[:2000]
    # pandas_df = get_df("pandas")[:2000]
    # benchmark_by_cases(polars_df, pandas_df)

    polars_df = get_df_by_trace_length("polars")
    pandas_df = get_df_by_trace_length("pandas")
    benchmark_by_trace_lengths(polars_df, pandas_df)

    # polars_bpi_dataframes = get_all_bpi("polars")
    # pandas_bpi_dataframes = get_all_bpi("pandas")
    # for i in range(len(polars_bpi_dataframes)):
    #     polars_df = get_df_by_trace_length("polars", df=polars_bpi_dataframes[i])
    #     pandas_df = get_df_by_trace_length("pandas", df=pandas_bpi_dataframes[i])
    #     benchmark_by_trace_lengths(polars_df, pandas_df)


if __name__ == '__main__':
    main()
