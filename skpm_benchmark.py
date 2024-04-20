from processing import agg_pandas, agg_polars
from dataframe import get_df


def benchmark():
    pd_df = get_df(engine="pandas")
    pl_df = get_df(engine="polars")
    agg_pandas(pd_df)
    agg_polars(pl_df)
    print("----------------------------")
    # win_agg_pandas(pd_df.iloc[:200000])
    # win_agg_polars(pl_df)


if __name__ == '__main__':
    benchmark()

    #
    #     df = pd.DataFrame({
    #         elc.timestamp: np.arange(10),
    #         elc.activity: np.random.randint(0, 10, 10),
    #         elc.resource: np.random.randint(0, 3, 10),
    #         elc.case_id: np.random.randint(0, 3, 10),
    #     }).sort_values(by=[elc.case_id, elc.timestamp])
    #     df = pd.get_dummies(df, columns=[elc.activity, elc.resource], dtype=int)
    #     df = df.drop(elc.timestamp, axis=1)
    #     agg = Aggregation(engine="pandas").fit_transform(df)
    #     # print(infer_column_types(df))
    #     print(agg)
    #
    #     pl_df = pl.DataFrame({
    #         elc.timestamp: np.arange(10),
    #         elc.activity: np.random.randint(0, 10, 10),
    #         elc.resource: np.random.randint(0, 3, 10),
    #         elc.case_id: np.random.randint(0, 3, 10),
    #     }).sort(by=[elc.case_id, elc.timestamp])
    #     pl_df = pl_df.to_dummies(columns=[elc.activity, elc.resource])
    #     agg = Aggregation(engine="polars").fit_transform(pl_df)
    # #     print(infer_column_types(df))
    #
    #     print(agg)
    #
    #
