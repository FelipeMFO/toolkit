import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Plots():
    """AI is creating summary for Plots
    """

    def __init__(self) -> None:
        pass

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
