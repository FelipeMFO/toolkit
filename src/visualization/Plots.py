
  
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


#-------

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import math


class Plots():
    """Module that will be used for rendering the plots.
    """

    def __init__(self, count: bool = False) -> None:
        """Initialize the module to manage plots

        Args:
            count (bool, optional): Define if it counts the number of total
            plots or not. Defaults to False.
        """
        if self.n_count:
            self.n_count = 0

    def plot_normalized_spectrum(self, sample: np.ndarray,
                                 limit_kev: tuple = (0, 1000),
                                 title: str = "Normalized specturm") -> None:
        """Plot histgram of a normalized spectrum.

        Args:
            sample (np.ndarray): spectrum already normalized of element
        """
        if len(sample.shape) > 1:
            y = sample.reshape(sample.shape[-1])
        else:
            y = sample
        x = np.linspace(limit_kev[0], limit_kev[1], num=len(y))

        plt.title(title)
        plt.xlabel("Energy (keV)")
        plt.ylabel("Relative intensity")
        plt.plot(x, y, color="red")
        plt.show()

    def plot_normalized_spectrums(self, samples: list,
                                  spectrum_size: int = 2000,
                                  figsize: tuple = (16, 12),
                                  limit_kev: tuple = (0, 1000), ncols: int = 2,
                                  title: str = "Normalized spectrums",
                                  ELEMENTS: list = ['Am', 'Ba', 'Co',
                                                    'Cs', 'Eu']) -> None:
        """Plot histgram of a normalized spectrums arranged in subplots.

        Args:
            samples (list): list of sample of element spectrum
            figsize (tuple, optional): size of the plot. Defaults to (16, 12).
            limit_kev (tuple, optional): limit of the x axis.
            Defaults to (0, 1000).
            ncols (int, optional): number of columns. Defaults to 2.
            title (str, optional): title. Defaults to "Normalized spectrums".
            ELEMENTS (list, optional): elements that will be plotted.
            Defaults to ['Am', 'Ba', 'Co', 'Cs', 'Eu'].
        """
        if (len(samples) % ncols) != 0:
            nrows = (len(samples)//ncols) + 1
        else:
            nrows = len(samples)//ncols
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        fig.suptitle(title, fontsize=20)
        for element_i, ax in enumerate(axs.flat):
            try:
                sample = samples[element_i]
            except IndexError:
                break
            if len(sample.shape) > 1:
                y = sample.reshape(spectrum_size)
            else:
                y = sample
            x = np.linspace(limit_kev[0], limit_kev[1], num=len(y))
            ax.plot(x, y, color="red")
            ax.set_xlabel("Energy (keV)")
            ax.set_ylabel("Relative intensity")
            ax.set_title(f"{ELEMENTS[element_i]} spectrum")
        plt.tight_layout()
        plt.show()

    def plot_normalized_spectrum_log(self, sample: np.ndarray,
                                     limit_kev: tuple = (0, 1000),
                                     title: str = "Normalized specturm")\
            -> None:
        """Plot histgram of a normalized spectrum with log y scale.

        Args:
            sample (np.ndarray): sample of element spectrum.
            limit_kev (tuple, optional): limit of the x axis.
            Defaults to (0, 1000).
            title (str, optional): title. Defaults to "Normalized spectrums".
        """

        if len(sample.shape) > 1:
            y = sample.reshape(sample.shape[-1])
        else:
            y = sample
        x = np.linspace(limit_kev[0], limit_kev[1], num=len(y))

        plt.title(title)
        plt.xlabel("Energy(keV)")
        plt.ylabel("Relative intensity")
        plt.yscale("log")
        plt.plot(x, y, color="red")
        plt.show()

    def plot_spectrum(self, energy_array: np.ndarray) -> None:
        """Plot a simple spectrum histogram from np.ndarray

        Args:
            energy_array (np.ndarray): energy np array from a element
        """
        spectrum, bins = np.histogram(
            energy_array,
            range=(0, 1000),
            bins=3000)
        plt.plot(bins[:-1], spectrum)
        plt.xlabel("Energy (keV)")
        plt.ylabel("Counts")

    def plot_spectrums_normal_and_log_overlapping(self, elements: dict,
                                                  title: str,
                                                  limit_kev: tuple = (0, 1000),
                                                  figsize=(16, 12),
                                                  export_pdf=False) -> None:
        """Plot spectrus by their normal plot and log plot. Overlapping between
        the simulated and real example identified inside the `elements` dict.

        Args:
            elements (dict): dictionary with elements as keys and first element
            as the real spectrum value and second one with simulated spectrum.
            title (str, optional): title of the plot.
            limit_kev (tuple, optional): limit of the x axis.
            Defaults to (0, 1000).
            figsize (tuple, optional): size of the figure.
            Defaults to (16, 12).
            export_pdf (bool, optional): if want to export the figures to
            pdf file. Defaults to False.
        """
        matplotlib.rc('font', size=14)
        matplotlib.rc('axes', titlesize=14)
        ncols = 2
        nrows = len(elements)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        fig.suptitle(title, fontsize=20)
        for row, key in enumerate(elements.keys()):
            # normal
            ax = axs[row][0]
            y_real = elements[key][0].reshape(-1)
            y_sim = elements[key][1].reshape(-1)
            x = np.linspace(limit_kev[0], limit_kev[1], num=len(y_real))

            ax.plot(x, y_real, color="blue", label='Real')
            ax.plot(x, y_sim, color="red", label='Simulated')

            ax.set_xlabel("Energy(keV)")
            ax.set_ylabel("Relative intensity")
            ax.set_title(f"Normalized {key.capitalize()} spectrum")
            ax.legend(loc="upper right")
            ax.set_xlim(0, 200)

            # log
            ax = axs[row][1]
            ax.plot(x, y_real, color="blue", label='Real')
            ax.set_yscale('log')
            ax.plot(x, y_sim, color="red", label='Simulated')
            ax.set_yscale('log')

            ax.set_xlabel("Energy(keV)")
            ax.set_ylabel("Relative intensity (log scale)")
            ax.set_title(f"Log scale on {key.capitalize()} spectrum")
            ax.legend(loc="upper right")
            ax.set_xlim(0, 200)
        plt.tight_layout()
        if export_pdf:
            plt.savefig('../../data/plot_range.pdf')

    def plot_spectrums_normal_overlapping(self, elements: dict,
                                          title: str,
                                          limit_kev: tuple = (0, 1000),
                                          figsize=(16, 12),
                                          export_pdf=False) -> None:
        """Plot spectrus by their normal plot. Overlapping between
        the simulated and real example identified inside the `elements` dict.

        Args:
            elements (dict): dictionary with elements as keys and first element
            as the real spectrum value and second one with simulated spectrum.
            title (str, optional): title of the plot.
            limit_kev (tuple, optional): limit of the x axis.
            Defaults to (0, 1000).
            figsize (tuple, optional): size of the figure.
            Defaults to (16, 12).
            export_pdf (bool, optional): if want to export the figures to
            pdf file. Defaults to False.
        """
        matplotlib.rc('font', size=14)
        matplotlib.rc('axes', titlesize=14)
        ncols = 1
        nrows = len(elements)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        fig.suptitle(title, fontsize=20)
        for row, key in enumerate(elements.keys()):
            # normal
            ax = axs[row]
            y_real = elements[key][0].reshape(-1)
            y_sim = elements[key][1].reshape(-1)
            x = np.linspace(limit_kev[0], limit_kev[1], num=len(y_real))

            ax.plot(x, y_real, color="blue", label='Real')
            ax.plot(x, y_sim, color="red", label='Simulated')

            ax.set_xlabel("Energy(keV)")
            ax.set_ylabel("Relative intensity")
            ax.set_title(f"Normalized {key.capitalize()} spectrum")
            ax.legend(loc="upper right")
            ax.set_xlim(0, 150)

        plt.tight_layout()
        if export_pdf:
            plt.savefig('../../data/plot_range_1.jpg')

    def plot_spectrums(
        self, energies_dict: dict, title: str,
        figsize: tuple = (16, 12), ncols: int = 2,
        max_energy: int = 350
    ) -> None:
        """Plot many simple spectrum histogram from a dictionary of elements.

        Args:
            energies_dict (dict):  energy dict from a element.
            title (str, optional): title of the plot.
            figsize (tuple, optional): size of the figure. Defaults to
            (16, 12)
            ncols (int, optional): number of columns. Defaults to 2.
            max_energy (int, optional): Maximum value of x axis.
            Defaults to 350.
        """
        if (len(energies_dict) % ncols) != 0:
            nrows = (len(energies_dict)//ncols) + 1
        else:
            nrows = len(energies_dict)//ncols
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        fig.suptitle(title, fontsize=20)
        for key, ax in zip(energies_dict.keys(), axs.flat):
            data = energies_dict[key]
            spectre, bins = np.histogram(
                data, range=(0, max_energy), bins=3000)

            ax.plot(bins[:-1], spectre)
            ax.set_xlabel("Energy (keV)")
            ax.set_ylabel("Counts")
            ax.set_title(f"Single events from {key.capitalize()} spectrum")
        plt.tight_layout()

    def plot_spectrums_overlapping(
        self, energies_dict: dict, title: str,
        figsize: tuple = (16, 12)
    ) -> None:
        """Plot spectrums overlapped over each other in a single plot.

        Args:
            energies_dict (dict): dictionary with key as element and value as
            the energies
            title (str): Plot's title
            figsize (tuple, optional): Size of figure. Defaults to (16,12).
        """

        plt.figure(figsize=figsize)
        plt.title(title)
        for key in energies_dict.keys():
            spectre, bins = np.histogram(
                energies_dict[key], range=(0, 202), bins=3000)

            plt.plot(bins[:-1], spectre)
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")

    def plot_histplots_with_thresholds(
        self,
        df: pd.DataFrame,
        column_to_loc: str,
        thresholds: dict,
        bins: int = 3000,
        title: str = "Histplots",
        figsize: tuple = (15, 15),
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
                    ax=ax, kde=True, bins=bins
                )
                ax.set_title(sub_plot_name, fontsize=20)
            else:
                sns.histplot(
                    df[column_to_loc].loc[
                        (df[column_to_loc] > threshold["lower_thresh"]) &
                        (df[column_to_loc] < threshold["upper_thresh"])
                    ],
                    ax=ax, kde=True, bins=bins
                )
                ax.set_title(sub_plot_name, fontsize=20)

        fig.suptitle(title, fontsize=40)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if export_pdf:
            plt.savefig(f'{kwargs["pdf_title"]}.pdf')

### MAB

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.pre_processing.Processing import Processing


class Plots():
    """[summary]
    """

    def __init__(self):
        self.processing = Processing()

    def plot_simulations(self, results: list) -> None:
        """[summary]

        Args:
            results (list): List with result from mcs.run() methdod,
            contains 4 objects, respectively: number_of_simulation,
            number_of_pulls, factor_importance_each_arm and cumulative_reward.

        """

        plt.ylim(0.0, 1.0)
        plt.title('Arms evolution')
        plt.xlabel('Time')
        plt.ylabel('Probability of select an arm')
        for i in range(results[2].shape[1]):
            plt.plot(range(results[1]), results[2][:, i], label=str(i))
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_cumulative_rewards(self, algs_results: tuple,
                                algs_names: tuple) -> None:
        """Plot cumulative rewards over time from received
        algorithm results.

        Args:
            algs_results (tuple): tuple of N algorithm results.
            algs_names (tuple): tuple of names of N algorithms.
        """
        plt.title('Cumulative rewards')
        plt.xlabel('Time')
        plt.ylabel('Rewards')
        for i in range(len(algs_results)):
            plt.plot(range(algs_results[i][1]),
                     algs_results[i][3],
                     label=algs_names[i])
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_algs_simulations_and_rewards(self, ctrs: dict,
                                          number_of_pulls: int,
                                          number_of_simulation: int,
                                          size: tuple = (18, 12)) -> None:
        """Plot 3 evolutions, one for each algorithm and plot in the end
        the cumulative reward.

        Args:
            ctrs (dict): [description]
            number_of_pulls (int): [description]
            number_of_simulation (int): [description]
        """
        ALGS = {
            "ths": "Thompson Sampling",
            "ucb1": "UCB1",
            "tuned": "UCB Tuned"
        }

        gs = gridspec.GridSpec(2, 3)
        fig = plt.figure(figsize=size)
        results = {}

        # Printing 3 algs on 1st row
        for i, alg in enumerate(ALGS):
            results[alg] = self.processing.get_simulations(ctrs, 
                                                           number_of_pulls,
                                                           number_of_simulation,
                                                           alg)
            ax = fig.add_subplot(gs[0, i])

            ax.set_title(f'Arms evolution {ALGS[alg]}.')
            ax.set_xlabel('Time (impressions)')
            ax.set_ylabel('Arm selection probability')
            for i in range(results[alg][2].shape[1]):
                ax.plot(range(results[alg][1]),
                        results[alg][2][:, i],
                        label=str(i))
            ax.legend()
            ax.grid(True)

        # Printing Rewards on 2nd row
        ax_big = fig.add_subplot(gs[1, :])
        ax_big.set_title('Cumulative rewards')
        ax_big.set_xlabel('Time (impressions)')
        ax_big.set_ylabel('Rewards (clicks)')
        for res in results:
            ax_big.plot(range(results[res][1]),
                        results[res][3],
                        label=ALGS[res])
        ax_big.legend()
        ax_big.grid(True)

##### K-Means Clustering

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer

from ..modeling.Modeling import Modeling
from ..metrics.Evaluations import Evaluations


class Plots():
    """Class responsible for plotting the figures of the project.
    """

    def __init__(self):
        self.modeling = Modeling()
        self.evaluations = Evaluations()

    def print_elbow_curves(self, pca_features_dict: dict,
                           title: str,
                           size: tuple = (18, 12)) -> None:
        """Receives a dictionary with values as the features
        to be clusterized in format of numpy.ndarray then it
        clusters those values and obtains a elbow curve optimal point
        following the score metric, and prints the graphic.

        Args:
            pca_features_dict (dict): dictionary with keys as the name of
            the data sets and values as numpy array with features
            used to cluster.
            title (str): Main title of figure.
            size (tuple, optional): Size of the figure. Defaults to (18, 12).
        """
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=size)
        axis_flat = axs.flatten()
        fig.suptitle(title, fontsize=20)

        visualgrid = [
            KElbowVisualizer(KMeans(), k=(3, 12), ax=axs[0][0], timings=False),
            KElbowVisualizer(KMeans(), k=(3, 12), ax=axs[0][1], timings=False),
            KElbowVisualizer(KMeans(), k=(3, 12), ax=axs[1][0], timings=False),
            KElbowVisualizer(KMeans(), k=(3, 12), ax=axs[1][1], timings=False),
            KElbowVisualizer(KMeans(), k=(3, 12), ax=axs[2][0], timings=False),
            KElbowVisualizer(KMeans(), k=(3, 12), ax=axs[2][1], timings=False)
        ]
        for i, key in enumerate(pca_features_dict):
            visualgrid[i].fit(pca_features_dict[key])
            axis_flat[i].set_title(key)
            axis_flat[i].set_xlabel('Number of clusters')
            axis_flat[i].set_ylabel('Distortion score')

        plt.tight_layout()
        plt.show()

    def print_evaluation_metrics(self, df: pd.DataFrame,
                                 title: str,
                                 clusters_range: tuple = (6, 8),
                                 size: tuple = (24, 20),
                                 use_inputed_model: bool = False,
                                 **kwargs) -> None:
        """Receives a DataFrames that will be clustered and analysed.
        Then after each clusterization for each K inside clusters_range,
        it plots the 3 metrics corresponding each clusterization. As well as
        returns the amount of objects per clusters.
        Column 1 is the level of importance, column 2 is the Silhoutte and
        COlumn 3 is scatter plot.


        Args:
            df (pd.DataFrame): DataFrame that will be clusterized for each
            K inside clusters_range.
            title (str): Main title of figure.
            clusters_range (list, optional): Range of Ks in which data will be
            clusterized and printed. Also the len of it is the number of
            columns. Defaults to [6,8].
            size (tuple, optional): Size of the figure. Defaults to (18, 12).
            use_inputed_model (bool, optional): Flag indicating if the graphics
            will be plotted considering an inputed algorithm instance as a
            kwarg.

        Kwargs:
            model (object): instance of object of clusterization algorithm.
        """
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=size)
        fig.suptitle(title, fontsize=20)
        col_title = {}

        for i, k in enumerate(range(clusters_range[0], clusters_range[1]+1)):
            # Clusterization
            if use_inputed_model is True:
                model = kwargs['model']
                ans = model(df, k)
            else:
                ans = self.modeling.quantile_transform_model(df, k)

            # Intercluster distance
            intercluster_vis = InterclusterDistance(ans["model"],
                                                    ax=axs[0][i],
                                                    random_state=42,
                                                    legend=True)
            intercluster_vis.fit(ans["features"])
            axs[0][i].set_title(f"{title} Intercluster distance K = {k}")
            axs[0][i].set_xlabel('Embeded C1')
            axs[0][i].set_ylabel('Embeded C2')

            # Silhoutte
            silhoutte_vis = SilhouetteVisualizer(ans["model"],
                                                 ax=axs[1][i],
                                                 colors='yellowbrick')
            silhoutte_vis.fit(ans["features"])
            axs[1][i].set_title(f"{title} Silhoutte plot K = {k}")
            axs[1][i].set_xlabel('Silhouette coefficient')
            axs[1][i].set_ylabel('Cluster label')

            # Scatterplot
            axs[2][i] = self.get_scatter_subplot_clusters(
                k_clusters=k,
                subplot=axs[2][i],
                model=ans["model"],
                y_pred=ans["clusters"],
                pca_features=ans["features"]
            )
            axs[2][i].set_title(f"{title} Scatterplot K = {k}")
            axs[2][i].set_xlabel('PC1')
            axs[2][i].set_ylabel('PC2')

            col_title[str(k)] = self.evaluations.get_clusters_and_amount(
                ans["data_frame"],
                ans["clusters"]
                )

        fig.text(0.15, 0.93, f"""Distribution for k=6:
Abs values:{col_title['6'][0]}
Percentage: {col_title['6'][1]}""", ha='left', va='top')
        fig.text(0.42, 0.93, f"""Distribution for k=7:
Abs values: {col_title['7'][0]}
Percentage: {col_title['7'][1]}""", ha='left', va='top')
        fig.text(0.69, 0.93, f"""Distribution for k=8:
Abs values: {col_title['8'][0]}
Percentage: {col_title['8'][1]}""", ha='left', va='top')

        plt.show()

    def get_scatter_subplot_clusters(self,
                                     k_clusters: int,
                                     subplot: plt.subplot,
                                     model: object,
                                     y_pred: np.ndarray,
                                     pca_features: np.ndarray) -> plt:
        """
        Generate scatter plots from the main principal
        components used to cluster data. Receives results
        of clusterization, as well as model and features used.
        Then receives a matplotlib.subplot object
        and embeds on it a scatter plot of clusters objects.

        Args:
            k_clusters (int): the amount of clusters.
            subplot (plt.subplot): subplot that will receive
            the scatter plot.
            model (object): model used to cluster.
            y_pred (np.ndarray): numpy array with clusters.
            pca_features (np.ndarray): numpy array with features
            used to cluster.

        Returns:
            matplotlib.pyplot : object containing the
            plots embeded on it.
        """
        _COLORS = ['red', 'blue', 'yellow', 'green', 'brown',
                   'gray', 'purple', 'cyan', 'olive']
        for k in range(k_clusters):
            subplot.scatter(pca_features[y_pred == k, 0],
                            pca_features[y_pred == k, 1],
                            s=100,
                            color=_COLORS[k],
                            label=f'cluster{k+1}')

        subplot.scatter(model.cluster_centers_[:, 0],
                        model.cluster_centers_[:, 1], s=100,
                        color='black', label='centroid')
        subplot.legend()

        return subplot
