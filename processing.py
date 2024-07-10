import pandas as pd
import polars as pl
from skpm.encoding import Aggregation

from helpers import timing


@timing
def agg_pandas(df: pd.DataFrame):
    rp = Aggregation().set_output(transform="pandas")
    rp.fit(df)
    out = rp.transform(df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == df.shape[0]


@timing
def agg_polars(df: pl.DataFrame):
    rp = Aggregation(engine="polars").set_output(transform="pandas")
    rp.fit(df)
    out = rp.transform(df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == df.shape[0]


@timing
def win_agg_pandas(df: pd.DataFrame):
    rp = Aggregation(window_size=3, num_method="sum", cat_method="sum", engine="pandas").set_output(
        transform="pandas")
    rp.fit(df)
    out = rp.transform(df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == df.shape[0]


@timing
def win_agg_polars(df: pl.DataFrame):
    rp = Aggregation(window_size=3, num_method="sum", cat_method="sum", engine="polars").set_output(
        transform="pandas")
    rp.fit(df)
    out = rp.transform(df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == df.shape[0]
