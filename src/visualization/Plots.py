
  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


class Plots():
    """Plot class used on Notebooks.
    """

    def __init__(self) -> None:
        pass

    def plot_barplot_per_row(self,
                             dfs: dict,
                             df_rows: dict,
                             title: str = "Barplots",
                             figsize: tuple = (24, 8),
                             *args) -> None:
        """Plots barplots per DataFrames' rows.
        Args:
            df_list (dict): dictionary listing DataFrames used
            per row.
            rows (dict): Dictionary containing the DataFrame colums that
            will be used as rows of the plt object.
            title (str, optional): Title of the plot. Defaults to "Barplots".
            figsize (tuple, optional): size of the figure. Defaults to (24, 8).
        """
        fig, axs = plt.subplots(
            nrows=len(dfs),
            ncols=len(df_rows),
            figsize=figsize
        )
        for col_i, (df_name, df) in enumerate(dfs.items()):
            for row_i, (row_name, row) in enumerate(df_rows.items()):
                temp_ax = axs[col_i][row_i]
                sns.barplot(
                    x='index',
                    y=df.transpose().reset_index()[row],
                    data=df.transpose().reset_index(),
                    ax=temp_ax
                )
                self.set_barplot_config(temp_ax, row_name, df_name, row_i)
        fig.suptitle(title, fontsize=40)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if args:
            plt.savefig(f'{args[0]}.pdf')

    def set_barplot_config(self,
                           ax: plt.subplot,
                           row_name: str,
                           df_name: str,
                           row_i: int) -> None:
        """Set barplot configuration for plot_barplot_per_row method.
        Args:
            ax (matplotlib.axes._subplots.AxesSubplot): axes object.
            that will be inserted and edited.
            row_name (str): name of the row on the figure plot.
            df_name (str): name of the DataFrame that is beign plotted.
            row_i (int): index of the row that is being edited.
        """
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=40, ha="right"
        )
        ax.set_xlabel('')
        ax.set_title(row_name, fontsize=20)
        if row_i == 0:
            ax.set_ylabel(df_name, rotation=90, fontsize=30)
        else:
            ax.get_yaxis().set_visible(False)

    def plot_histplots_with_thresholds(
        self,
        df: pd.DataFrame,
        column_to_loc: str,
        thresholds: dict,
        title: str = "Histplots",
        figsize: tuple = (15, 30),
        export_pdf: bool = False,
        **kwargs
    ) -> None:
        """Plots histplots of a DataFrame filtering using thresholds established
        on a provided dict.
        Args:
            df (pd.DataFrame): DataFrame with data to be plotted.
            column_to_loc (str): column to be analyzed on histplots.
            title (str, optional): figure title. Defaults to "Histplots".
            figsize (tuple, optional): figure size. Defaults to (24, 8).
            export_pdf (bool, optional): True to export image, False to
            only plot. Defaults to False.
        """
        plots = len(thresholds)
        fig, axs = plt.subplots(
            nrows=math.ceil(plots/2),
            ncols=2,
            figsize=figsize,
        )
        for zipped in zip(axs.flat, thresholds.items()):
            ax, sub_plot_name, threshold = (zipped[0],
                                            zipped[1][0],
                                            zipped[1][1])
            if threshold["include_thresh"]:
                sns.histplot(
                    df[column_to_loc].loc[
                        (df[column_to_loc] >= threshold["lower_thresh"]) &
                        (df[column_to_loc] <= threshold["upper_thresh"])
                    ],
                    ax=ax, kde=True
                )
                ax.set_title(sub_plot_name, fontsize=20)
            else:
                sns.histplot(
                    df[column_to_loc].loc[
                        (df[column_to_loc] > threshold["lower_thresh"]) &
                        (df[column_to_loc] < threshold["upper_thresh"])
                    ],
                    ax=ax, kde=True
                )
                ax.set_title(sub_plot_name, fontsize=20)

        fig.suptitle(title, fontsize=40)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if export_pdf:
            plt.savefig(f'{kwargs["pdf_title"]}.pdf')