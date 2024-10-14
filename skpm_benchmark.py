import pandas as pd
import polars as pl

from dataframe import get_df, get_time_and_memory, plot_timings, get_stratified_df_by_trace_length, \
    get_df_percentage_by_sklearn, \
    plot_memories, get_all_bpi, get_time_and_memory_by_trace_length, get_df_by_trace_length, plot_timings_bar, \
    plot_timings_grouped_bars
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


def benchmark_by_trace_lengths(polars_df: pl.DataFrame, pandas_df: pd.DataFrame, df_name):
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
                 pandas_win_agg_times, x_label='Length of trace', filename=df_name)
    # Plot Memory
    plot_memories(lengths_pandas, total, polars_agg_memory, pandas_agg_memory, polars_win_agg_memory,
                  pandas_win_agg_memory, x_label='Length of trace', filename=df_name)


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


def BPI_benchmark(polars_df, df_name):
    percent = 0.02
    rounds = 3
    bars = 16
    grouped_timings = [[[] for _ in range(rounds)] for _ in range(bars)]
    timings = [0 for _ in range(bars)]
    for i in range(rounds):
        polars_df_stratified = get_stratified_df_by_trace_length("polars", df=polars_df, size=percent,
                                                                 random_state=42 + i)
        pandas_df_stratified = polars_df_stratified.to_pandas()
        pl_times_1, pd_times_1 = benchmark_by_trace_lengths_by_percent(polars_df_stratified,
                                                                       pandas_df_stratified)

        polars_df_not_stratified = get_stratified_df_by_trace_length("polars", df=polars_df,
                                                                     size=percent, stratify=False,
                                                                     random_state=None)
        pandas_df_not_stratified = polars_df_not_stratified.to_pandas()
        pl_times_no_stratified_2, pd_times_no_stratified_2 = benchmark_by_trace_lengths_by_percent(
            polars_df_not_stratified,
            pandas_df_not_stratified)

        for j in range(4):
            base_index = 4 * j
            timings[base_index] += pl_times_1[j]
            timings[base_index + 1] += pl_times_no_stratified_2[j]
            timings[base_index + 2] += pd_times_1[j]
            timings[base_index + 3] += pd_times_no_stratified_2[j]

    plot_timings_bar(['PL 25%', 'NS PL 25%', 'PD 25%', 'NS PD 25%',
                      'PL 50%', 'NS PL 50%', 'PD 50%', 'NS PD 50%',
                      'PL 75%', 'NS PL 75%', 'PD 75%', 'NS PD 75%',
                      'PL 100%', 'NS PL 100%', 'PD 100%', 'NS PD 100%'
                      ], timings, df_name
                     )

    # grouped version
    # for j in range(4):
    #     base_index = 4 * j
    #     grouped_timings[base_index][i] = pl_times_1[j]
    #     grouped_timings[base_index + 1][i] = pl_times_no_stratified_2[j]
    #     grouped_timings[base_index + 2][i] = pd_times_1[j]
    #     grouped_timings[base_index + 3][i] = pd_times_no_stratified_2[j]
    #
    # plot_timings_grouped_bars([i for i in range(1, rounds + 1)], grouped_timings,
    #                           ['Polars 25%', 'Polars 25% (No Stratified)', 'Pandas 25%', 'Pandas 25% (No Stratified)',
    #                            'Polars 50%', 'Polars 50% (No Stratified)', 'Pandas 50%', 'Pandas 50% (No Stratified)',
    #                            'Polars 75%', 'Polars 75% (No Stratified)', 'Pandas 75%', 'Pandas 75% (No Stratified)',
    #                            'Polars 100%', 'Polars 100% (No Stratified)', 'Pandas 100%',
    #                            'Pandas 100% (No Stratified)'
    #                            ]
    #                           )


def run_BPI_benchmark():
    polars_bpi_dataframes, names = get_all_bpi("polars")
    for i, df in enumerate(polars_bpi_dataframes):
        name = names[i]
        # BPI_benchmark(df, name)
        try:
            BPI_benchmark(df, name)
        except Exception as e:
            print(e)


def main():
    # polars_df = get_df("polars")[:2000]
    # pandas_df = get_df("pandas")[:2000]
    # benchmark_by_cases(polars_df, pandas_df)

    run_BPI_benchmark()

    # percent = 0.02
    # polars_df = get_df_by_trace_length("polars")
    # # pandas_df = get_df_by_trace_length("pandas")
    # polars_df = polars_df.head(int((len(polars_df) * (percent * 100)) / 100))
    # pandas_df = polars_df.to_pandas()
    # # pandas_df = pandas_df.head(int((len(pandas_df) * (percent * 100)) / 100))
    # benchmark_by_trace_lengths(polars_df, pandas_df)

    # polars_bpi_dataframes, names = get_all_bpi("polars")
    # # pandas_bpi_dataframes = get_all_bpi("pandas")
    # for i in range(len(polars_bpi_dataframes)):
    #     df = polars_bpi_dataframes[i]
    #     if names[i] == "BPI19":
    #         df = df[:500000]
    #     else:
    #         continue
    #     polars_df = get_df_by_trace_length("polars", df=df)
    #     pandas_df = get_df_by_trace_length("pandas", df=df)
    #     benchmark_by_trace_lengths(polars_df, pandas_df, names[i])


if __name__ == '__main__':
    main()
