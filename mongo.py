"""module that accesses MongoDB"""
import json
import logging
import os
import pymongo
import pymongoarrow.monkey
from dotenv import load_dotenv

load_dotenv()

# pymongoarrow.monkey.patch_all()


class MongoClient:
    """class that accesses MongoDB"""

    def __init__(self):
        """initiates pymongo client"""
        logging.info("Initiating Mongo client ...")

        uri = (
            f"mongodb+srv://{os.getenv('MONGO_USER')}:{os.getenv('MONGO_PWD')}@"
            f"{os.getenv('MONGO_CLUSTER')}.mpakn.mongodb.net/?retryWrites=true&w=majority"
        )
        self.client = pymongo.MongoClient(uri)
        self.collection = None

        logging.info("Mongo client initiated successfully.")

    def set_collection(self, database, collection):
        """sets the collection that the client will accessed"""
        logging.info("Accessing collection: %s.%s", database, collection)
        self.collection = self.client[database][collection]
