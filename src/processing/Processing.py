import pandas as pd


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
