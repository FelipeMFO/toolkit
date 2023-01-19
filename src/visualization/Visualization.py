import pandas as pd
import copy
from IPython.display import display_html
from itertools import chain, cycle


def front(self, n=5):
    return self.iloc[:, :n]


def back(self, n=5):
    return self.iloc[:, -n:]


def display_columns(self, n=5):
    return pd.concat([self.front(n), self.back(n)], axis=1)


def display_rows(self, n=5):
    return pd.concat([self.head(n), self.tail(n)])


def display(self, i=5, j=5):
    return self.display_columns(i).display_rows(j)


pd.DataFrame.front = front
pd.DataFrame.back = back
pd.DataFrame.display_columns = display_columns
pd.DataFrame.display_rows = display_rows
pd.DataFrame.display = display


class Visualization():
    """Class with methods for visualization of DataFrames.
    """

    def __init__(self):
        pass

    def visualize_percentiles(
        self, df: pd.DataFrame,
        column: str, percentile_fraction: float
    ) -> pd.DataFrame:
        """Filter DataFrame by the percentile of the given column.

        Args:
            df (pd.DataFrame): DataFrame to be filtered.
            column (str): column that will be used to filter values.
            percentile_fraction (float): percentile value that will
            be used as threshold to filter data.

        Returns:
            pd.DataFrame: filtered DataFrame.
        """
        total_quantiles = len(range(int(1/percentile_fraction)))
        df_ans = pd.DataFrame({
            str(fraction/total_quantiles): [
                df.quantile(fraction/total_quantiles)[column]
            ] for fraction in range(total_quantiles)
        })

        df_ans = df_ans.transpose()
        df_ans.columns = [
            f"Percentiles of {percentile_fraction} of '{column}' column"
        ]
        return df_ans

    def display_side_by_side(self, *args, titles=cycle([''])):
        """Prints the given DataFrames showing their number of rows.
        Receive arguments as a dictionary.

        Args:
            titles ([type], optional): titles of the DataFrames provided.
            Defaults to cycle(['']).
        """
        html_str = ''
        for df, title in zip(args, chain(titles, cycle(['</br>']))):
            df_ = copy.copy(df.display())
            html_str += \
                '<th style="text-align:center"><td style="vertical-align:top">'
            html_str += f'<h2>{title}</h2>'
            html_str += \
                df_.to_html().replace('table', 'table style="display:inline"')
            html_str += '</td></th><br>'
            html_str += f"{len(df)} rows x {len(df.columns)} columns"
        display_html(html_str, raw=True)

    def display_vertically(self, *args, titles=cycle([''])):
        """Prints the given DataFrames showing their number of rows.
        Receive arguments as a dictionary.

        Args:
            titles ([type], optional): titles of the DataFrames provided.
            Defaults to cycle(['']).
        """
        html_str = ''
        for df, title in zip(args, chain(titles, cycle(['</br>']))):
            df_ = copy.copy(df.display())
            html_str += \
                '<th style="text-align:center"><td style="vertical-align:top">'
            html_str += f'<h3>{title}</h3>'
            html_str += \
                df_.to_html().replace('table', 'table style="display:inline"')
            html_str += '</td></th><br>'
            html_str += f"{len(df)} rows x {len(df.columns)} columns"
        display_html(html_str, raw=True)
