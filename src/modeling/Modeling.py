<<<<<<< HEAD
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
=======
# -------------------- Initialization and Saving Functions --------------------


def split(df, validation_ratio):
    X = df.loc[:, "f1":"REDSHIFT_SPEC"]
    y = df.loc[:, "REDSHIFT_SPEC"]
    train, test = h2o.H2OFrame(X).split_frame(ratios=[validation_ratio], seed=1)

    return X, y, train, test


def auto_ML(df, n_models, validation_ratio=0.5):
    """
    Initialize h2o, a new Auto ML object and already train it applying verification following the validation ratio.
    """
    h2o.init(ip="localhost", port=54323)
    aml = H2OAutoML(max_models=n_models, seed=1)
    X, y, train, test = split(df, validation_ratio)

    aml.train(
        x=list(df.loc[:, "f1":"f20"].columns),
        y="REDSHIFT_SPEC",
        training_frame=train,
        leaderboard_frame=test,
    )

    lb = aml.leaderboard
    print("Leaderboard: ", lb.head(rows=lb.nrows), "\n")
    print("Leader: ", aml.leader, "\n")

    return aml


def save_aml_models(aml):
    """
    Save all aml object models on the respective folder.
    """
    aml_pd = aml.leaderboard.as_data_frame(use_pandas=True)
    lb = aml.leaderboard
    path = f"../../models/mojo_{len(lb)-2}_ensemble/"

    for i in range(len(lb)):
        model = h2o.get_model(aml_pd["model_id"][i])
        model.save_mojo(path, force=True)
        print(f"Model {aml_pd['model_id'][i]} saved.")


# -------------------- Prediction Related Functions --------------------


def read_models(path):
    """
    Return dict of models inside specified folder.
    """
    for (dirpath, dirnames, filenames) in walk(path):
        pass
    models = {}
    for file in filenames:
        models[file] = h2o.import_mojo(f"../../models/mojo_50_ensemble/{file}")
    return models


def gen_predictions(df, model, validation_ratio=0.5, return_len=True):
    """
    Receives DataFrame and model, it split the frame and returns predictions and len of splits.
    """
    X, y, train, test = split(df, validation_ratio)
    preds = model.predict(h2o.H2OFrame(X.iloc[-len(test) :].loc[:, "f1":"f20"]))
    print(
        "\n",
        f"{len(train)} train/test objects ",
        "\n",
        f"and {len(test)} validation objects",
    )
    if return_len:
        return preds, len(train), len(test)
    return preds


def gen_models_predictions(df, models, validation_ratio=0.5):
    """
    Receives DataFrame, dict fo models and returns a DataFrame with N + 1 columns.
    N is the number of models on dict (including Stacked Ensemble) plus the true value as first column (REDSHIFT_SPEC).
    """
    X, y, train, test = split(df, validation_ratio)

    df_predictions = pd.DataFrame(X.iloc[-len(test) :]["REDSHIFT_SPEC"])
    df_predictions.reset_index(inplace=True)
    for model_name in models.keys():
        preds = gen_predictions(
            df, models[model_name], validation_ratio=validation_ratio, return_len=False
        )
        df_ = preds.as_data_frame()
        df_.columns = [model_name]
        df_predictions = pd.concat([df_predictions, df_], axis=1)

    df_predictions.set_index("ID", inplace=True)
    return df_predictions


def gen_gaussian_kde(preds, points=100):
    """
    Return gaussian_kde values after receiving the prediction values.
    """
    pdf = gaussian_kde(preds).pdf(np.linspace(preds.min(), preds.max(), points))
    return pdf, np.linspace(preds.min(), preds.max(), points)


def print_gaussian_kde(pdf, preds, real_value, best_pred, title, points=100):
    """
    Receives the Probability Density Function generated by gaussian KDE method and prints it
    with the best prediction, real value and a histogram distribution.
    """
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    axs.axvline(best_pred, c="k", ls="--", lw="3", label="Best prediction")
    axs.axvline(real_value, c="r", ls="--", lw="3", label="Real value")
    axs.hist(preds, density=True, histtype="step", label="Models prediction histogram")
    axs.legend(loc="best")
    axs.set_title(title)
    axs.plot(np.linspace(preds.min(), preds.max(), points), pdf)


def gen_df_gaussian_kde(df_predictions, export_csv=False, **kwargs):
    ans = copy.deepcopy(df_predictions)
    predictions = df_predictions.loc[
        :, df_predictions.columns[1] : df_predictions.columns[-1]
    ].values.reshape(len(df_predictions), 52)
    ans["PDF"], ans["PDF_X_axis"], ans["predictions"] = None, None, None
    for i in range(len(predictions)):
        ans["PDF"].iloc[i], ans["PDF_X_axis"].iloc[i] = gen_gaussian_kde(predictions[i])
        ans["predictions"].iloc[i] = predictions[i]
    if export_csv:
        ans[
            [
                "REDSHIFT_SPEC",
                kwargs.get("best_model"),
                "PDF",
                "PDF_X_axis",
                "predictions",
            ]
        ].to_csv(kwargs.get("path"))
    return ans[
        ["REDSHIFT_SPEC", kwargs.get("best_model"), "PDF", "PDF_X_axis", "predictions"]
    ]
>>>>>>> 869bda484b51fd7ffca540027b1967fca27342a4
