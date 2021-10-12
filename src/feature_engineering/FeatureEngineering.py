from datetime import timedelta

import pandas as pd


class FeatureEngineering():
    """Class of feature engineering.
    """

    def __init__(self) -> None:
        pass

    def gen_empty_df_datetime_index(
            self, datetime_interval: tuple,
            columns: list, minutes: int) -> pd.DataFrame:
        """Receives datetime_interval and generates DataFrame with
        n minutes per row as index.

        Args:
            datetime_interval (tuple): tuple of datetime.datetime objects.
            Beginning and ending.
            columns (list): columns of the created DataFrame.

        Returns:
            pd.DataFrame: DataFrame with index as datetime.datetime
            and all values set to NaN.
        """
        def datetime_range(start, end, delta):
            current = start
            while current < end:
                yield current
                current += delta
        dts = [
            dt.strftime('%Y-%m-%d %H:%M:%S') for dt in
            datetime_range(
                datetime_interval[0],
                datetime_interval[1],
                timedelta(minutes=minutes))
        ]
        df = pd.DataFrame(index=dts,
                          columns=columns)
        df.index = pd.to_datetime(df.index)
        df.index.name = 'time'
        return df

    def get_shrinked_columns(self, df:pd.DataFrame) -> pd.DataFrame:
        """Shrink the columns of the dataframe .

        Args:
            df (pd.DataFrame): [description]

        Returns:
            pd.DataFrame: [description]
        """
        # Turn the columns to an index and drop the old one
        df = df.stack().reset_index(level=0, drop=True)
        # Turn the values of each column into lists and transpose the result
        df.groupby(df.index).apply(list).to_frame().transpose()
        return df