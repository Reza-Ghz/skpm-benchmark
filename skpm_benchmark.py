from helpers import percentage
from processing import agg_pandas, agg_polars, win_agg_polars, win_agg_pandas
from dataframe import get_df, get_timings, plot_timings


def benchmark():
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


if __name__ == '__main__':
    benchmark()
