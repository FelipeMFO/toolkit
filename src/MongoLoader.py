from pymongo import MongoClient
from pymongo.collection import ReturnDocument


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
        """Insert one row into specified collection
        for valuesllowing document format.

        Args:
            collection(pymongo.collection.Collection):
            Collection in which
            will be inserted.
            document(dict): dictionary with attributes (as dict_keys)
            and values (as dict_items).

        Returns:
            doc_id(bson.objectid.ObjectId): Inserted object ID.
        """
        doc_id = collection.insert_one(document).inserted_id
        return doc_id

    def find_document(self, collection, filters):
        """Find documentcorresponding filters conditions.

        Args:
            collection(pymongo.collection.Collection): Collection
            in which will be inserted.
            filters(dict): dictionary with attributes and
            values as verification.

        Returns:
            list of found elements.
        """
        return list(collection.find(filters))

    def update_document(self, collection, filters, document):
        """Update row into specified collection with document
        values following filters conditions.

        Args:
            collection(pymongo.collection.Collection): Collection
            in which will be inserted.
            filters(dict): dictionary with attributes and values
            as verification.
            document(dict): dictionary with attributes (as dict_keys)
            and values (as dict_items).
        """
        collection.update_one(filters, {"$set": document}, upsert=True)

    def update_doc_and_return(
        self, collection, filters, set_values, inc_values, return_value=True
    ):
        """Update row into specified collection with document values
        following filters conditions.

        Args:
            collection(pymongo.collection.Collection): Collection in
            which will be inserted.
            filters(dict): dictionary with attributes and values as
             verification.
            set_values(dict): dictionary with attributes (as dict_keys)
             and values (as dict_items) to be set.
            inc_values(dict): dictionary with attributes and values
            to be incremented on the ones existent on Collection.

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
