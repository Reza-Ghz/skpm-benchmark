import pandas as pd
import polars as pl

from dataframe import get_df, get_time_and_memory, plot_timings, get_stratified_df_by_trace_length, \
    get_df_percentage_by_sklearn, \
    plot_memories, get_all_bpi, get_time_and_memory_by_trace_length, get_df_by_trace_length, plot_timings_bar
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

    lengths_polars = polars_df.select(pl.col('length').unique()).to_series().sort()
    lengths_pandas = sorted(pandas_df['length'].unique())

    # Get timings for Polars and Pandas
    polars_timings, polars_memory = get_time_and_memory_by_trace_length(polars_df, agg_polars, win_agg_polars,
                                                                        lengths_polars)
    pandas_timings, pandas_memory = get_time_and_memory_by_trace_length(pandas_df, agg_pandas, win_agg_pandas,
                                                                        lengths_pandas)

    # Extract timings for plotting
    polars_agg_times = [polars_timings[l][0] for l in lengths_polars]
    polars_win_agg_times = [polars_timings[l][1] for l in lengths_polars]
    pandas_agg_times = [pandas_timings[l][0] for l in lengths_pandas]
    pandas_win_agg_times = [pandas_timings[l][1] for l in lengths_pandas]

    # Extract timings for plotting
    polars_agg_memory = [polars_memory[l][0] for l in lengths_polars]
    polars_win_agg_memory = [polars_memory[l][1] for l in lengths_polars]
    pandas_agg_memory = [pandas_memory[l][0] for l in lengths_pandas]
    pandas_win_agg_memory = [pandas_memory[l][1] for l in lengths_pandas]
    # TODO: change x label from percentage -> X should be the trace length
    # TODO: start writing down all the steps and explain how each memory works
    # TODO: define what is the benchamark by each metric, case, trace lengh and ...
    # Plot timings
    plot_timings(lengths_polars, total, polars_agg_times, pandas_agg_times, polars_win_agg_times,
                 pandas_win_agg_times, x_label='Length of trace')
    # Plot Memory
    plot_memories(lengths_pandas, total, polars_agg_memory, pandas_agg_memory, polars_win_agg_memory,
                  pandas_win_agg_memory, x_label='Length of trace')


def benchmark_by_trace_lengths_by_percent(polars_df: pl.DataFrame, pandas_df: pd.DataFrame):
    lengths_polars = polars_df.select(pl.col('length').unique()).to_series().sort()
    start = lengths_polars.min()
    end = lengths_polars.max()
    percentages = [0.25, 0.50, 0.75, 1]
    values = [start + p * (end - start) for p in percentages]
    lengths_polars = values
    lengths_pandas = values

    polars_timings, polars_memory = get_time_and_memory_by_trace_length(polars_df, agg_polars, lambda x: x,
                                                                        lengths_polars)
    pandas_timings, pandas_memory = get_time_and_memory_by_trace_length(pandas_df, agg_pandas, lambda x: x,
                                                                        lengths_pandas)
    polars_agg_times = [polars_timings[l][0] for l in lengths_polars]
    pandas_agg_times = [pandas_timings[l][0] for l in lengths_pandas]
    return polars_agg_times, pandas_agg_times


def main():
    # polars_df = get_df("polars")[:2000]
    # pandas_df = get_df("pandas")[:2000]
    # benchmark_by_cases(polars_df, pandas_df)

    percent = 0.01
    timings = [[[] for _ in range(3)] for _ in range(8)]

    for i in range(3):
        polars_df_stratified = get_stratified_df_by_trace_length("polars", size=percent,
                                                                 random_state=42 + i)
        pandas_df_stratified = polars_df_stratified.to_pandas()
        pl_times_1, pd_times_1 = benchmark_by_trace_lengths_by_percent(polars_df_stratified,
                                                                       pandas_df_stratified)

        polars_df_stratified = get_stratified_df_by_trace_length("polars", size=percent, stratify=False,
                                                                 random_state=42 + i)
        pandas_df_stratified = polars_df_stratified.to_pandas()
        pl_times_2, pd_times_2 = benchmark_by_trace_lengths_by_percent(polars_df_stratified,
                                                                       pandas_df_stratified)
        timings[0][i] = pl_times_1[0]
        timings[1][i] = pd_times_1[0]
        timings[2][i] = pl_times_1[1]
        timings[3][i] = pd_times_1[1]
        timings[4][i] = pl_times_1[2]
        timings[5][i] = pd_times_1[2]
        timings[6][i] = pl_times_1[3]
        timings[7][i] = pd_times_1[3]

    plot_timings_bar([1, 2, 3], timings,
                     ['Polars 25%', ' Pandas 25%', 'Polars 50%', ' Pandas 50%', 'Polars 75%', ' Pandas 75%',
                      'Polars 100%', ' Pandas 100%'])

    # polars_df = get_df_by_trace_length("polars")
    # # pandas_df = get_df_by_trace_length("pandas")
    # polars_df = polars_df.head(int((len(polars_df) * (percent * 100)) / 100))
    # pandas_df = polars_df.to_pandas()
    # # pandas_df = pandas_df.head(int((len(pandas_df) * (percent * 100)) / 100))
    # benchmark_by_trace_lengths(polars_df, pandas_df)

    # polars_bpi_dataframes = get_all_bpi("polars")
    # pandas_bpi_dataframes = get_all_bpi("pandas")
    # for i in range(len(polars_bpi_dataframes)):
    #     polars_df = get_df_by_trace_length("polars", df=polars_bpi_dataframes[i])
    #     pandas_df = get_df_by_trace_length("pandas", df=pandas_bpi_dataframes[i])
    #     benchmark_by_trace_lengths(polars_df, pandas_df)


if __name__ == '__main__':
    main()
