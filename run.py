# built-in imports
import logging
from pprint import pprint

# internal imports
from mongo import MongoClient

# third-party inports
import pandas as pd

NAME = "Demo für Mohammad"

logging.basicConfig(
    level=logging.INFO,
    format=f"[%(asctime)s] {NAME} - %(levelname)s: %(message)s",
    filename="logging.log",
    filemode="w",
)

MONGO = MongoClient()


if __name__ == "__main__":

    MONGO.set_collection("mohammadDB", "wohnung_kaufen")
    cursor = MONGO.collection.aggregate(
        pipeline=[
            {
                "$match": {
                    "AP_community": "München",
                    "realEstate.numberOfRooms": 1.5,
                }
            },
            {
                "$project": {
                    "postcode": "$postcode",
                    "observationDate": "$observationDate",
                    "title": "$realEstate.title",
                    "rooms": "$realEstate.numberOfRooms",
                    "price": "$realEstate.price.value",
                    "Mohammads_Wohnraum": "$realEstate.livingSpace",
                }
            },
        ]
    )

    test_data = pd.DataFrame.from_records(list(cursor))
    print(test_data)
    print(test_data.info())
    print(test_data.describe())

    # alle inserate die in 1.5 bis zum 7.5 online waren

    for doc in cursor:
        # logging.info("Reading record with source_id %s", doc["source_id"])
        pprint(doc)
        break
