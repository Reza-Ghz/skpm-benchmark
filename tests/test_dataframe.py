import pytest

from dataframe import get_df, get_df_percentage
from helpers import percentage



def test_dataframe():
    polars_df = get_df("polars")
    polars_df_25 = get_df_percentage(polars_df, 25)
    polars_df_50 = get_df_percentage(polars_df, 50)
    polars_df_75 = get_df_percentage(polars_df, 75)
    polars_df_100 = get_df_percentage(polars_df, 100)
    total, cols = polars_df.shape
    assert polars_df_25.shape == (percentage(25, total), cols)
    assert polars_df_50.shape == (percentage(50, total), cols)
    assert polars_df_75.shape == (percentage(75, total), cols)
    assert polars_df_100.shape == (total, cols)

    pandas_df = get_df("pandas")
    pandas_df_25 = get_df_percentage(pandas_df, 25)
    pandas_df_50 = get_df_percentage(pandas_df, 50)
    pandas_df_75 = get_df_percentage(pandas_df, 75)
    pandas_df_100 = get_df_percentage(pandas_df, 100)
    total, cols = polars_df.shape
    assert pandas_df_25.shape == (percentage(25, total), cols)
    assert pandas_df_50.shape == (percentage(50, total), cols)
    assert pandas_df_75.shape == (percentage(75, total), cols)
    assert pandas_df_100.shape == (total, cols)
