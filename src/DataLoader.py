import os
from .GoogleUtils import GoogleUtils
from .processing.Processing import Processing
from .utils import read_file
import pandas as pd
from astropy.io import fits
from pymongo import MongoClient
import time
import pandas as pd


os.chdir(os.path.dirname(__file__))


class DatasetLoader():
    """
    This class loads datasets used in the analysis.
    From Pickles, queries or standard loads frequently made on notebooks.
    """

    def __init__(self, folder: str) -> None:
        self.folder = folder
        self.subdirectories = [x[0] for x in os.walk(folder)][1:]
        self.files = next(os.walk(folder), (None, None, []))[2]
        self.subdirectories_path = \
            [self.folder + subdirectory for subdirectory in self.subdirectories]
        self.files_path = \
            [self.folder + file for file in self.files]
        self.env = os.getenv("ENV", "dev")
        # self.google_utils = GoogleUtils()
        self.processing = Processing()

    def get_df_from_query(self, query: str) -> pd.DataFrame:
        """Returns pd DataFrame from SQL query on string format."""
        raw_dataset = self.google_utils.read_from_bq(query)
        return raw_dataset

    def load_query(self,
                   query: str,
                   online: bool = True,
                   force_to_query: bool = False,
                   update_dataframe: bool = False,
                   **kwargs) -> pd.DataFrame:
        """
        Args:
            query (str): name of the query file on 'queries/' directory.
            online (bool, optional): True to check on GCS and query on BQ if
            not present on GCS. False to only check on data/ dir.
            Defaults to True.
            force_to_query (bool, optional): True to force a BQ query process.
            Defaults to False.
            update_dataframe(bool): True to save the forced query on GCS and
            local. Defaults to True.
        Kwargs:
            BUCKET (str): name of the bucket on GCP to read and save.
            FOLDER (str): folder path of the bucket on GCP to read and save.
        Returns:
            pd.DataFrame: DataFrame resulted of specified query.
        """
        query_content = read_file(f'queries/{query}.sql')

        # Forcing query and refresh.
        if force_to_query is True:
            df = self.google_utils.read_from_bq(query_content)
            if update_dataframe is True:
                try:
                    self.google_utils.upload_df(
                        df,
                        kwargs['BUCKET'],
                        f"{kwargs['FOLDER']}{query}.pkl"
                    )
                except KeyError:
                    return ("Please inform bucket and folder \
to save DataFrame.")
                else:
                    return df
            else:
                return df
        # Normal flow.
        # # Online false.
        if online is False:
            try:
                df = open(f'../data/{query}.pkl', 'rb')
            except IOError:
                print('File not found on local data directory.',
                      'Please connect to GCP to query on BQ or download from \
GCS')
            else:
                return pd.read_pickle(df)

        # # Online true.
        else:
            try:
                df = self.load_df_pickle(
                    kwargs['BUCKET'],
                    f"{kwargs['FOLDER']}{query}.pkl"
                )
            except AttributeError:
                print('No pickle found on GCP, starting query on BQ.')
            except KeyError:
                return ('Please inform GCP bucket and folder or\
change `online` to FALSE or `force_to_query` to TRUE')
            else:
                df.to_pickle(f"../data/{query}.pkl")
                print(f'File {kwargs["BUCKET"]}.{kwargs["FOLDER"]}.{query} \
downloaded. Returning DataFrame and saving local pickle.')
                return df

            df = self.google_utils.read_from_bq(query_content)
            self.google_utils.upload_df(
                df,
                kwargs['BUCKET'],
                f"{kwargs['FOLDER']}{query}.pkl"
            )
            df.to_pickle(f"../data/{query}.pkl")
            print("Query loaded on BQ, file saved on GCP and local pickle.")
            return df

    def load_df_pickle(self, bucket_name: str,
                       destination_blob_name: str) -> pd.DataFrame:
        """Load pickle from a bucket and blob on GCP."""
        self.google_utils.download_from_bucket(bucket_name,
                                               destination_blob_name,
                                               'df_temp.pkl')
        df = pd.read_pickle('df_temp.pkl')
        os.remove('df_temp.pkl')
        return df

    def load_xlsx(self, file_name: str) -> dict:
        """Loads an xlsx file.
        Args:
            file_name (str): file to be read.
        Returns:
            dict: dictionary with all sheets (tabs).
        """
        xl_file = pd.ExcelFile(f"../data/{file_name}.xlsx")
        dfs = {
            sheet_name: xl_file.parse(sheet_name)
            for sheet_name in xl_file.sheet_names
        }
        return dfs

    def load_all_xlsx(self) -> dict:
        """Load all xlsx files and return a dict with them.
        Returns:
            dict: dictionnary with all files that will be used on
            analysis loaded.
        """
        xlsx = {
            "missing": self.load_xlsx("missing_hotels_in_hpa"),
            "it_br": self.load_xlsx("hurb_popular_itinerary_queries_br"),
            "it_us": self.load_xlsx("hurb_popular_itinerary_queries_us"),
            "it_v2": self.load_xlsx("hurb_popular_itinerary_v2"),
            "it_v3": self.load_xlsx("hurb_popular_itinerary_v3")
        }

        return xlsx

    def load_fits(self, remove_na: bool = True) -> dict:
        """Load astropy fits type and transforms it into a
        dictionary.

        Args:
            remove_na (bool, optional): Once Na was an element
            detected using other sensors, there's a flag to remove
            it if wanted. Defaults to True.

        Returns:
            dict: dictionary with file name on lower case as key and
            fits objects properties as values structured as another
            dict.
        """
        ans = {
            f"{file.replace('.fits', '').lower()}": {
                "header": fits.open(self.folder+file)[1].header,
                "tabdata": fits.open(self.folder+file)[1].data
            }
            for file in self.files
        }
        if remove_na:
            del ans["na_recalibrated_touslespix"]
        return ans


class MongoConnector():
    """Connector class to implement Mongo DB handle methods.

    Attributes:
        client (pymongo.mongo_client.MongoClient): Mongo client.
        db (pymongo.database.Database): Selected Mongo Database.
    """

    def __init__(self, path="localhost",
                 client="general", host="27017"):
        """Initialize client and db.

        Args:
            client(str): Client to be accessed. Set as default as 'general'.
        """
        self.path = path
        self.client = MongoClient(f"mongodb://{self.path}:{host}/")
        self.db = self.client[client]

    def get_collection(self, collection_name):
        """Collection getter.
        Args:
            collection(str): Collection to be returned.

        Returns:
            pymongo.collection.Collection: Collection object.
        """
        return self.db[collection_name]

    def insert_document(self, collection, document):
        """Insert one row into specified collection fovaluesllowing document format.

        Args:
            collection(pymongo.collection.Collection): Collection in which will be inserted.
            document(dict): dictionary with attributes (as dict_keys) and values (as dict_items).

        Returns:
            doc_id(bson.objectid.ObjectId): Inserted object ID.
        """
        doc_id = collection.insert_one(document).inserted_id
        return doc_id

    def find_document(self, collection, filters):
        """Find documentcorresponding filters conditions.

        Args:
            collection(pymongo.collection.Collection): Collection in which will be inserted.
            filters(dict): dictionary with attributes and values as verification.

        Returns:
            list of found elements.
        """
        return list(collection.find(filters))

    def update_document(self, collection, filters, document):
        """Update row into specified collection with document values following filters conditions.

        Args:
            collection(pymongo.collection.Collection): Collection in which will be inserted.
            filters(dict): dictionary with attributes and values as verification.
            document(dict): dictionary with attributes (as dict_keys) and values (as dict_items).
        """
        collection.update_one(filters, {"$set": document}, upsert=True)

    def update_doc_and_return(
        self, collection, filters, set_values, inc_values, return_value=True
    ):
        """Update row into specified collection with document values following filters conditions.

        Args:
            collection(pymongo.collection.Collection): Collection in which will be inserted.
            filters(dict): dictionary with attributes and values as verification.
            set_values(dict): dictionary with attributes (as dict_keys) and values (as dict_items) to be set.
            inc_values(dict): dictionary with attributes and values to be incremented on the ones existent on Collection.

        Returns:
            ans(dict): dictionary with updated incremented and set values.
        """
        ans = collection.find_one_and_update(
            filters,
            {"$set": set_values, "$inc": inc_values},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        if return_value:
            return ans
