import pandas as pd
import numpy as np
import copy
import re


class Processing():
    """Processes processing .
    """

    def __init__(self) -> None:
        pass

    def drop_rows_by_occurrence(
        self, df: pd.DataFrame,
        column: str, threshold: int,
        type: str = "less"
    ) -> pd.DataFrame:
        """Drop rows by occurrence of a given occurrence .

        Args:
            df (pd.DataFrame): [description]
            column (str): [description]
            value (int): [description]
            type (str, optional): [description]. Defaults to "less".

        Returns:
            pd.DataFrame: [description]
        """
        if type == "less":
            df_ans = df[df[column].map(df[column].value_counts()) > threshold]
        else:
            df_ans = df[df[column].map(df[column].value_counts()) < threshold]

        return df_ans

    def save_pickles(self, dfs_dict: dict) -> None:
        """Save into pickles DataFrames from dicts loaded from sheets of xlsx format.

        Args:
            dfs_dict (dict): DataFrame dictionary that will provide the names
            of files to be saved as keys and the files as values.
        """
        for table in dfs_dict:
            for sheet in dfs_dict[table]:
                sheet_name = "_".join(
                    re.sub(r'\b\w{1,3}\b', '', sheet.replace("-", "")).split()
                ).lower()
                dfs_dict[table][sheet].to_pickle(
                    f"../data/df_{table}_{sheet_name}.pkl"
                )

    def get_quantiles_cut(
        self, df: pd.DataFrame,
        column: str,
        quantile_down: float,
        quantile_up: float
    ) -> pd.DataFrame:
        """Return a DataFrame with the results between the quantiles provided.

        Args:
            df (pd.DataFrame): DataFrame that will be filtered.
            column (str): column that will be used to filter DataFrame.
            quantile_down (float): lower threshold.
            quantile_up (float): upper threshold.

        Returns:
            pd.DataFrame: [description]
        """

        df_ans = copy.copy(df)
        df_ans = df_ans.loc[
            (df_ans[column] >= df_ans.quantile(quantile_down)[column]) &
            (df_ans[column] <= df_ans.quantile(quantile_up)[column])
        ]
        return df_ans

    def get_merged_df(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        key_column: str,
        columns_to_drop: list
    ) -> pd.DataFrame:
        """Merge the dataframes into a new dataframe on key_column and
        removes duplicates. Merge as inner.

        Args:
            df1 (pd.DataFrame): first DataFrame.
            df2 (pd.DataFrame): second DataFrame.
            key_column (str): column to be used to join as key.
            columns_to_drop (list): columns desired to drop.

        Returns:
            pd.DataFrame: DataFrame merged, without duplicates.
        """
        df_ans = pd.merge(
            df1,
            df2,
            how='inner',
            on=[key_column]
        )
        df_ans = df_ans.drop(
            columns=columns_to_drop
        ).drop_duplicates()

        return df_ans

    def get_filter_df_by_list(
        self, df: pd.DataFrame,
        column: str,
        list_to_filter: np.array,
        include: bool
    ) -> pd.DataFrame:
        """Returns a DataFrame with the given list of values
        filtered by the given list of values .

        Args:
            df (pd.DataFrame): DataFrame that will be filtered.
            column (str): column used to filter.
            list_to_filter (np.array): list with elements to filter.
            include (bool): True to include the elements on the list,
            False to exlude.

        Returns:
            pd.DataFrame: [description]
        """
        df_ans = copy.copy(df)
        if include:
            df_ans = df[include & (df[column].isin(
                list_to_filter))
            ].reset_index().drop(columns=["index"])
        else:
            df_ans = df[~include & (df[column].isin(
                list_to_filter))
            ].reset_index().drop(columns=["index"])
        return df_ans

    def get_ordered_columns_list(
        self,
        df: pd.DataFrame,
        index: str
    ) -> dict:
        """Returns a dict of DataFrames of columns ordered by the index.
        Removing values with 0.

        Args:
            df (pd.DataFrame): DataFrame to be sorted
            index (str): column to be set as index.

        Returns:
            dict: dictionary with all columns, one per key, where
            the values are new DataFrames containing only that column
            and sorted.
        """
        df_copy = copy.copy(df)
        df_copy.set_index(index, inplace=True)

        ans = {
            column: df_copy[[column]].loc[
                ~(df_copy[[column]] == 0.0).all(axis=1)
            ].sort_values(
                column,
                axis=0,
                ascending=False) for column in df_copy.columns
        }

        return ans
