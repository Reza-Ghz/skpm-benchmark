import polars as pl
from skpm.config import EventLogConfig as elc


def get_df(engine="polars"):
    df = (pl.read_ndjson("./ts-events.json")
          .rename({"ts": elc.timestamp, "id": elc.case_id, "event": elc.activity})
          .with_columns(pl.from_epoch(elc.timestamp, time_unit="s"))
          .sort(by=[elc.case_id, elc.timestamp]))
    df = df.to_dummies(columns=[elc.activity])
    df = df.drop(elc.timestamp)
    return df if engine == "polars" else df.to_pandas() if engine == "pandas" else None

# TODO: create a function to get 25 50 75 100% of variants in event log
