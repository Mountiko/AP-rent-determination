# built-in imports
import logging

# internal imports
from mongo import MongoClient


NAME = "Demo für Mohammad"

logging.basicConfig(
    level=logging.INFO,
    format=f"[%(asctime)s] {NAME} - %(levelname)s: %(message)s",
    filename="logging.log",
    filemode="w",
)

MONGO = MongoClient()


if __name__ == "__main__":

    MONGO.set_collection("mohammadDB", "wohnung_mieten")
    cursor = MONGO.collection.find()

    for doc in cursor:
        logging.info("Reading record with source_id %s", doc["source_id"])
