from helpers import percentage
from processing import agg_pandas, agg_polars, win_agg_polars, win_agg_pandas
from dataframe import get_df, get_timings, plot_timings, get_df_by_trace_length, get_df_percentage_by_sklearn


def benchmark_by_cases():
    polars_df = get_df("polars")[:10000]
    pandas_df = get_df("pandas")[:10000]
    total = len(polars_df)
    percentages = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    # percentage_labels = [f'{p}({percentage(p, len(polars_df))})' for p in percentages]

    # Get timings for Polars and Pandas
    polars_timings = get_timings(polars_df, agg_polars, win_agg_polars, percentages)
    pandas_timings = get_timings(pandas_df, agg_pandas, win_agg_pandas, percentages)

    # Extract timings for plotting
    polars_agg_times = [polars_timings[p][0] for p in percentages]
    polars_win_agg_times = [polars_timings[p][1] for p in percentages]
    pandas_agg_times = [pandas_timings[p][0] for p in percentages]
    pandas_win_agg_times = [pandas_timings[p][1] for p in percentages]

    # Plot timings
    plot_timings(percentages, total, polars_agg_times, pandas_agg_times, polars_win_agg_times,
                 pandas_win_agg_times)


def benchmark_by_trace_lengths():
    pandas_df = get_df_by_trace_length("pandas")
    polars_df = get_df_by_trace_length("polars")
    total = len(polars_df)
    percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                   0.95, 1.0]

    # Get timings for Polars and Pandas
    polars_timings = get_timings(polars_df, agg_polars, win_agg_polars, percentages, get_df_percentage_by_sklearn)
    pandas_timings = get_timings(pandas_df, agg_pandas, win_agg_pandas, percentages, get_df_percentage_by_sklearn)

    # Extract timings for plotting
    polars_agg_times = [polars_timings[p][0] for p in percentages]
    polars_win_agg_times = [polars_timings[p][1] for p in percentages]
    pandas_agg_times = [pandas_timings[p][0] for p in percentages]
    pandas_win_agg_times = [pandas_timings[p][1] for p in percentages]

    # Plot timings
    plot_timings(percentages, total, polars_agg_times, pandas_agg_times, polars_win_agg_times,
                 pandas_win_agg_times)


if __name__ == '__main__':
    benchmark_by_cases()
    benchmark_by_trace_lengths()
