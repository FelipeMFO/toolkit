from google.cloud import bigquery
from google.cloud import storage
import pandas as pd
import os


class GoogleUtils():
    """
    Helps with Gooogle Cloud operations
    Args:
        env: enviroment defiens if the class will use local key credentials or
        online credentials
        auth_token_path: path to token to use in local authentication
    """
    def __init__(self):
        self.project_id = os.getenv('GCP_PROJECT_NAME', 'bi-data-science')
        self.init_clients()
        self.default_job_config = bigquery.job.QueryJobConfig(
                use_legacy_sql=False,
                use_query_cache=True
        )

    def init_clients(self):
        """
        Init Google Cloud Plataform client services
        """
        self.bq_client = bigquery.Client()

    def read_from_bq(self, query):
        """
        Executes a query on BigQuery and return the result as a pandas
        dataframe
        Args:
            query: query to execute (SQL Standard)
        Returns:
            query_result: query result as a Pandas Dataframe
        """
        query_job = self.bq_client.query(query, self.default_job_config)
        query_result = query_job.result().to_dataframe()
        return query_result

    def upload_to_bucket(self, bucket_name: str,
                         source_file_name: str,
                         destination_blob_name: str):
        """Uploads a file to the bucket."""

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(
            "File uploaded to {} inside {} bucket.".format(
                destination_blob_name, bucket_name
            )
        )

    def download_from_bucket(self, bucket_name: str,
                             destination_blob_name: str,
                             source_file_name: str):
        """Downloads a file from the bucket.
        Args:
            bucket_name (str): bucket destination on GCP.
            destination_blob_name (str): directory path on GCP.
        """
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.get_blob(destination_blob_name)
        blob.download_to_filename(source_file_name)

    def upload_df(self,
                  df: pd.DataFrame,
                  bucket_name: str,
                  destination_blob_name: str):
        """Upload DataFrame to google cloud storage.
        Args:
            df (pd.DataFrame): DataFrame to be exported to GCP.
            bucket_name (str): bucket destination on GCP.
            destination_blob_name (str): directory path on GCP.
        """
        df.to_pickle('df_temp.pkl')
        self.upload_to_bucket(bucket_name,
                              'df_temp.pkl',
                              destination_blob_name)

        os.remove('df_temp.pkl')