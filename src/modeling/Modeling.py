from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import h2o
from h2o.estimators import H2OKMeansEstimator


class Modeling():
    """Class with functions responsible for clustering models.
    """

    def __init__(self):
        pass

    def get_clusterization_features(self, df: pd.DataFrame,
                                    pca_dataset_representation: float = 0.95,
                                    get_variance_ratio: bool = False)\
            -> np.ndarray:
        """Applies quantile trasnformer and PCA set to represent
        95% of data.

        Args:
            df (pd.DataFrame): DataFrame that will be processed.
            pca_dataset_representation (float, optional): percentage of the
            amount of data that shall be represented after PCA. Defaults
            to 0.95.

        Returns:
            np.ndarray: Array with features to be cluster. If
            get_variance_ratio is True, return feature representation
            importance to dataset.
        """
        _df = df.copy()
        _df.replace(np.NaN, 0, inplace=True)
        scaler = preprocessing.QuantileTransformer()
        df_scaled = scaler.fit_transform(_df)
        pca_95 = PCA(pca_dataset_representation)
        X = pca_95.fit_transform(df_scaled)

        if get_variance_ratio is False:
            return X
        else:
            return X, pca_95.explained_variance_ratio_

    def quantile_transform_model(self, df: pd.DataFrame,
                                 n_clusters: int,
                                 pca_dataset_representation: float = 0.95) \
            -> dict:
        """Returns the best model clusterization answers. The answers
        are structured as a dictionary with the features used to cluster,
        the clusters itself, the DataFrame, the variance ration of the
        features before clustering, and the object as a KMeans model.

        Args:
            df (pd.DataFrame): DataFrame that will be clustered.
            n_clusters (int): Number of K clusters on KMeans.
            pca_dataset_representation (float): Option to consider the
            representation of features before clustering. Defaults to 0.95.

        Returns:
            dict: Dictionary with answers structured with keys to return
            features, clusters, data_frame, variance_rate and model.
        """
        df_ans = df.copy()
        df_ans.replace(np.NaN, 0, inplace=True)

        X, variance_ratio = \
            self.get_clusterization_features(df_ans,
                                             get_variance_ratio=True)

        model = KMeans(n_clusters=n_clusters, random_state=42)
        y_pred = model.fit_predict(X)
        df_ans["bucket"] = y_pred
        cols = [list(df_ans.columns)[-1]] + list(df_ans.columns)[:-1]

        return {
            "features": X,
            "clusters": y_pred,
            "data_frame": df_ans[cols],
            "variance_ratio": variance_ratio,
            "model": model
        }

    def old_model(self, df: pd.DataFrame,
                  n_clusters: int) -> dict:
        """First MVP model of the Trivago clusterization problem.
        Returns a dictionary with the same keys: features, clusters,
        data_frame and model.

        Args:
            df (pd.DataFrame): DataFrame that will be clustered.
            n_clusters (int): Number of K clusters on KMeans.
            pca_dataset_representation (float): Option to consider the
            representation of features before clustering. Defaults to 0.95..

        Returns:
           dict: Dictionary with answers structured with keys to return
           features, clusters, data_frame and model.
        """

        df_ans = df.copy()
        df_ans.replace(np.inf, 0, inplace=True)

        # PCA
        n_PCAs = 8
        pca = PCA(n_components=n_PCAs)
        pca.fit(df_ans.fillna(0).values)
        pca_values = pca.transform(df_ans.fillna(0).values)

        # Robust Scaler
        scaler = preprocessing.RobustScaler()
        df_scaled = scaler.fit_transform(pca_values)
        df_scaled = pd.DataFrame(pca_values, index=df_ans.index,
                                 columns=['PCA1', 'PCA2', 'PCA3', 'PCA4',
                                          'PCA5', 'PCA6', 'PCA7', 'PCA8'
                                          ])
        # MinMax Scaler
        scaler_mM = preprocessing.MinMaxScaler()
        df_scaled_twice = scaler_mM.fit_transform(df_scaled)
        df_scaled_twice = pd.DataFrame(df_scaled_twice,
                                       index=df.index,
                                       columns=['PCA1', 'PCA2', 'PCA3', 'PCA4',
                                                'PCA5', 'PCA6', 'PCA7', 'PCA8'
                                                ])
        # K-Means
        X = df_scaled_twice.values
        model = KMeans(n_clusters=n_clusters, random_state=42)
        y_pred = model.fit_predict(X)

        df_ans["bucket"] = y_pred
        cols = [list(df_ans.columns)[-1]] + list(df_ans.columns)[:-1]

        return {
            "features": X,
            "clusters": y_pred,
            "data_frame": df_ans[cols],
            "model": model
        }

    def old_model_kpis(self, df: pd.DataFrame,
                       n_clusters: int) -> dict:
        """First MVP model using only kpis as input.

        Args:
            df (pd.DataFrame): DataFrame that will be clustered.
            n_clusters (int): Number of K clusters on KMeans.
            pca_dataset_representation (float): Option to consider the
            representation of features before clustering. Defaults to 0.95..

        Returns:
           dict: Dictionary with answers structured with keys to return
           features, clusters, data_frame and model.
        """

        df_ans = df.copy()
        df_ans.replace(np.inf, 0, inplace=True)

        # Robust Scaler
        scaler = preprocessing.RobustScaler()
        df_scaled = scaler.fit_transform(df_ans)

        # MinMax Scaler
        scaler_mM = preprocessing.MinMaxScaler()
        df_scaled_twice = scaler_mM.fit_transform(df_scaled)

        # K-Means
        X = df_scaled_twice
        model = KMeans(n_clusters=n_clusters, random_state=42)
        y_pred = model.fit_predict(X)

        df_ans["bucket"] = y_pred
        cols = [list(df_ans.columns)[-1]] + list(df_ans.columns)[:-1]

        return {
            "features": X,
            "clusters": y_pred,
            "data_frame": df_ans[cols],
            "model": model
        }
    
    def h2o_kmeans_model(self, features: np.ndarray,
                         n_clusters: int) -> dict:
        """Receives already the features pre processed and returns
        answers from H2o KMeans clusterization model
        Answer follows the same structure as other models with keys:
        features, clusters, data_frame and model.

        Args:
            features (np.ndarray): features already pre processed
            to be clusterized following the h2o KMeans algorithm.
            n_clusters (int): Number of K clusters on KMeans.

        Returns:
            dict: Dictionary with answers structured with keys to return
           features, clusters, data_frame and model.
        """
        h2o_df = h2o.H2OFrame(pd.DataFrame(features))
        predictors = list(pd.DataFrame(features).columns)
        train, valid = h2o_df.split_frame(ratios=[.8], seed=42)
        kmeans = H2OKMeansEstimator(k=n_clusters,
                                    estimate_k=True,
                                    standardize=False,
                                    seed=42)
        kmeans.train(x=predictors,
                     training_frame=train,
                     validation_frame=valid)

        all_pred = kmeans.predict(h2o_df)
        return {
            "features": features,
            "clusters": np.array(all_pred.as_data_frame()),
            "data_frame": pd.DataFrame(features),
            "model": kmeans
        }
