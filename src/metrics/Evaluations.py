import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

from yellowbrick.cluster import intercluster_distance
from yellowbrick.cluster import KElbowVisualizer

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Evaluations():
    """Class containing the methods to assess the models using specific
    evaluation metrics.
    """

    def __init__(self):
        pass

    def mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Returns MAPE"""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Returns RMSE"""
        return sqrt(mean_squared_error(y_true, y_pred))

    def get_mape_rmse(self, df: pd.DataFrame, kpi_a: str, kpi_b: str):
        """Return a print comparing RMSE and MAPE through 2 pd Series."""
        print(f'MAPE: {self.mape(df[kpi_a].values, df[kpi_b].values)}')
        print(f'RMSE: {self.rmse(df[kpi_a].values, df[kpi_b].values)}')

    def sorting_by_amount(self, y_pred: np.ndarray) -> np.ndarray:
        """Sorts the labels, 0 will be the bucket with most SKUs,
        1 the bucket with second most, and so on.

        Args:
            y_pred (np.array): Array with predictions, contains the
            number of the buckets.

        Returns:
             np.ndarray: clustered values ordered by amount.
             0 will be the one with more ocurrences, 1 the second
             most, and so on.
        """
        clusters, amount = np.unique(y_pred, return_counts=True)
        sorted_ = [x for _, x in sorted(zip(amount, clusters))]
        sorted_.reverse()
        conversion = dict(zip(sorted_, clusters))
        return np.vectorize(conversion.get)(y_pred)

    def get_clusters_and_amount(self, df: pd.DataFrame,
                                y_pred: np.ndarray) -> tuple:
        """Returns the clusters arranged by order of members
        and the percentage of that amount from the total points.

        Args:
            df (pd.DataFrame): DataFrame already clustered.
            y_pred (np.ndarray): Clustered values in order.

        Returns:
            tuple: First value as the absolute number of members in
            a cluster. Second value as its percentage.
        """
        abs_numbers = np.unique(self.sorting_by_amount(y_pred).tolist(),
                                return_counts=True)
        percetages = np.round(abs_numbers[1]/abs_numbers[1].sum(axis=0)*100,
                              decimals=1)
        return (abs_numbers[1], percetages)

    def get_intercluster_distance(self,
                                  n_clusters: int,
                                  pca_features: np.ndarray) -> None:
        """Inter cluster distance shows on an new embeding space
        how clusters are divided and their importance factor (size) of
        boobles. WARNING: Those numbers and sizes are representative on a
        new multidimensional scaling 2 dimensional space, not on the principal
        components received.
        For more information: https://www.scikit-yb.org/en/latest/api/cluster/icdm.html#:~:text=Intercluster%20distance%20maps%20display%20an,according%20to%20a%20scoring%20metric.
        The idea is to observe how close they are to each other (intercluster
        distance), and how big the boobles are (intracluster distance).
        It does not matter if the boobles intersect each other, it does not
        mean nothing.

        Args:
            n_clusters (int): number of clusters
            pca_features (np.ndarray): features used to cluster.
        """
        intercluster_distance(KMeans(n_clusters, random_state=42),
                              pca_features,
                              random_state=42,
                              legend=False)

    def get_elbow_curve(self, pca_features: np.ndarray) -> None:
        """Receives the features to cluster and then plots
        the elbow curve to find inside the range of 3-12 the best
        number to be K in the KMeans algorithm. It uses the distortion
        score as the metric to find the optimal elbow point.

        Args:
            pca_features (np.ndarray): features to be clusterized.
        """
        visualizer = KElbowVisualizer(KMeans(), k=(3, 12), timings=False)
        visualizer.fit(pca_features)
        visualizer.show()

    def get_silhoutte_curve(self,
                            features: np.ndarray,
                            y_pred: np.ndarray) -> None:
        """Receives the buckets clusterization values and
        the features on a space desired to calculate the
        silhoutte curve. Adapted code from scikit-learn example lib.

        Args:
            features (np.ndarray): features to be classified with silhooutte
            curve.
            y_pred (np.ndarray): clusterization values
        """
        n_clusters = len(np.unique(np.array(y_pred)))

        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(7, 7)

        ax1.set_xlim([-1, 1])
        ax1.set_ylim([0, len(features) + (n_clusters + 1) * 10])

        silhouette_avg = silhouette_score(
            features,
            y_pred.reshape(-1, 1).ravel()
        )
        sample_silhouette_values = silhouette_samples(
            features,
            y_pred.reshape(-1, 1).ravel()
        )

        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[y_pred == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(
            ("Silhouette analysis for KMeans clustering \
on sample data with n_clusters = %d" % n_clusters),
            fontsize=14, fontweight='bold'
        )

    plt.show()

    def get_scatter_plot_clusters(self,
                                  k_clusters: int,
                                  model: object,
                                  y_pred: np.ndarray,
                                  pca_features: np.ndarray) -> None:
        """Generate scatter plots from the main principal
        components used to cluster data. Receives results
        of clusterization, as well as model and features used.

        Args:
            k_clusters (int): the amount of clusters.
            model (object): model used to cluster.
            y_pred (np.ndarray): numpy array with clusters.
            pca_features (np.ndarray): numpy array with features
            used to cluster.
        """
        _COLORS = ['red', 'blue', 'yellow', 'green', 'brown',
                   'gray', 'purple', 'cyan', 'olive']
        for k in range(k_clusters):
            plt.scatter(pca_features[y_pred == k, 0],
                        pca_features[y_pred == k, 1],
                        s=100,
                        color=_COLORS[k],
                        label=f'cluster{k+1}')

        plt.scatter(model.cluster_centers_[:, 0],
                    model.cluster_centers_[:, 1], s=100,
                    color='black', label='centroid')
        plt.legend()
