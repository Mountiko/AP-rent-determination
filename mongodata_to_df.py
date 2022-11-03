# built-in imports
import logging
from pprint import pprint
import datetime

# internal imports
from mongo import MongoClient
import pymongo

# third-party imports
import pandas as pd
import numpy as np


NAME = "Demo für Mohammad"

logging.basicConfig(
    level=logging.INFO,
    format=f"[%(asctime)s] {NAME} - %(levelname)s: %(message)s",
    filename="logging.log",
    filemode="w",
)

MONGO = MongoClient()


if __name__ == "__main__":

    MONGO.set_collection("mohammadDB", "immonet_wohnung_mieten")

    # Sort stages
    stage_sort_by_obseravtionDate = {"$sort": {"observationDate": pymongo.DESCENDING}}

    # Project stages

    stage_projection_all = {
        "$project": {
            "observationDate": "$observationDate",
            "state": "$state",
            "city": "$city",
            "AP_community": "$AP_community",
            "livingSpace": "$livingArea",
            "roomCount": "$rooms",
            "floor": "$level",
            "constructionYear": "$constructionYear",
            "apartmentType": "$apartmentType",
            "condition": "$condition",
            "rentPerParking": "$rent.parking.rentPerParking",
            "deposit": "$rent.deposit",
            "base_rent": "$rent.rentBase",
            "propertyTransferTax": "$cost.propertyTransferTax",
            "parkingCount": "$rent.parking.parkingCount",
            "energyEfficiencyClass": "$energy.energyEfficiencyClass",
            "energyConsumption": "$energy.energyConsumption",
            "energyType": "$energy.energyType",
            "heatingType": "$energy.heatingType",
            "guestToilet": "$features.geustToilette",
            "balcony": "$features.balcony",
            "terrace": "$features.terrace",
            "garden": "$features.garden",
            "furnished": "$features.furnished",
            "builtInKitchen": "$features.builtInKitchen",
            "floorHeating": "$features.floorHeating",
            "floorType": "$features.floor",
            "cableConnection": "$features.cableConnection",
            "undergroundParking": "$features.undergroundParking",
            "outsideParking": "$features.parkingLot",
            "cellar": "$features.cellar",
            "lift": "$features.lift",
            "barrierfree": "$features.barrierfree",
            "wheelchair": "$features.wheelchair",
            "fireplace": "$features.fireplace",
            "longitude": "$coords.lng",
            "latitude": "$coords.lat",
            "coord_confidence": "$coords.confidence",
            "distance_kindergarden": "$location.education.distance_kindergarden_m",
            "distance_school": "$location.education.distance_school_m",
            "distance_university": "$location.education.distance_university_m",
            "distance_ATM": "$location.general_needs.distance_ATM_m",
            "distance_bakery": "$location.general_needs.distance_bakery_m",
            "distance_DIY_store": "$location.general_needs.distance_doityourself_store_m",
            "distance_hairdresser": "$location.general_needs.distance_hairdresser_m",
            "distance_supermarket": "$location.general_needs.distance_supermarket_m",
            "distance_clinic": "$location.health.distance_clinic_m",
            "distance_doctor": "$location.health.distance_doctor_m",
            "distance_hospital": "$location.health.distance_hospital_m",
            "distance_pharmacy": "$location.health.distance_pharmacy_m",
            "distance_airport": "$location.infrastructure.distance_airport_m",
            "distance_bus_stop": "$location.infrastructure.distance_bus_stop_m",
            "distance_charging_station": "$location.infrastructure.distance_charging_stations_m",
            "distance_fuel": "$location.infrastructure.distance_fuel_m",
            "distance_harbour": "$location.infrastructure.distance_harbours_m",
            "distance_motorway_junction": "$location.infrastructure.distance_motorway_juntion_m",
            "distance_recycling_center": "$location.infrastructure.distance_recycling_center_m",
            "distance_train_station": "$location.infrastructure.distance_train_station_m",
            "distance_tram_station": "$location.infrastructure.distance_tram_stations_m",
            "distance_bar": "$location.recreation.distance_bar_m",
            "distance_beergarden": "$location.recreation.distance_beergarden_m",
            "distance_nightclub": "$location.recreation.distance_nightclub_m",
            "distance_restaurant": "$location.recreation.distance_restaurant_m",
            "distance_children": "$location.social_facilities.distance_for_children_m",
            "distance_seniors": "$location.social_facilities.distance_for_senior_m",
            "distance_shelter": "$location.social_facilities.distance_shelter_m",
            "distance_hotel": "$location.tourism.distance_hotel_m",
            "distance_museum": "$location.tourism.distance_museum_m",
        }
    }

    stage_projection_object_spec = {
        "$project": {
            "observationDate": "$observationDate",
            "state": "$state",
            "city": "$city",
            "AP_community": "$AP_community",
            "livingSpace": "$livingArea",
            "roomCount": "$rooms",
            "floor": "$level",
            "constructionYear": "$constructionYear",
            "apartmentType": "$apartmentType",
            "condition": "$condition",
            "rentPerParking": "$rent.parking.rentPerParking",
            "deposit": "$rent.deposit",
            "base_rent": "$rent.rentBase",
            "propertyTransferTax": "$cost.propertyTransferTax",
            "parkingCount": "$rent.parking.parkingCount",
            "energyEfficiencyClass": "$energy.energyEfficiencyClass",
            "energyConsumption": "$energy.energyConsumption",
            "energyType": "$energy.energyType",
            "heatingType": "$energy.heatingType",
            "guestToilet": "$features.geustToilette",
            "balcony": "$features.balcony",
            "terrace": "$features.terrace",
            "garden": "$features.garden",
            "furnished": "$features.furnished",
            "builtInKitchen": "$features.builtInKitchen",
            "floorHeating": "$features.floorHeating",
            "floorType": "$features.floor",
            "cableConnection": "$features.cableConnection",
            "undergroundParking": "$features.undergroundParking",
            "outsideParking": "$features.parkingLot",
            "cellar": "$features.cellar",
            "lift": "$features.lift",
            "barrierfree": "$features.barrierfree",
            "wheelchair": "$features.wheelchair",
            "fireplace": "$features.fireplace",
        }
    }

    stage_project_location_spec = {
        "$project": {
            "longitude": "$coords.lng",
            "latitude": "$coords.lat",
            "coord_confidence": "$coords.confidence",
            "distance_kindergarden": "$location.education.distance_kindergarden_m",
            "distance_school": "$location.education.distance_school_m",
            "distance_university": "$location.education.distance_university_m",
            "distance_ATM": "$location.general_needs.distance_ATM_m",
            "distance_bakery": "$location.general_needs.distance_bakery_m",
            "distance_DIY_store": "$location.general_needs.distance_doityourself_store_m",
            "distance_hairdresser": "$location.general_needs.distance_hairdresser_m",
            "distance_supermarket": "$location.general_needs.distance_supermarket_m",
            "distance_clinic": "$location.health.distance_clinic_m",
            "distance_doctor": "$location.health.distance_doctor_m",
            "distance_hospital": "$location.health.distance_hospital_m",
            "distance_pharmacy": "$location.health.distance_pharmacy_m",
            "distance_airport": "$location.infrastructure.distance_airport_m",
            "distance_bus_stop": "$location.infrastructure.distance_bus_stop_m",
            "distance_charging_station": "$location.infrastructure.distance_charging_stations_m",
            "distance_fuel": "$location.infrastructure.distance_fuel_m",
            "distance_harbour": "$location.infrastructure.distance_harbours_m",
            "distance_motorway_junction": "$location.infrastructure.distance_motorway_juntion_m",
            "distance_recycling_center": "$location.infrastructure.distance_recycling_center_m",
            "distance_train_station": "$location.infrastructure.distance_train_station_m",
            "distance_tram_station": "$location.infrastructure.distance_tram_stations_m",
            "distance_bar": "$location.recreation.distance_bar_m",
            "distance_beergarden": "$location.recreation.distance_beergarden_m",
            "distance_nightclub": "$location.recreation.distance_nightclub_m",
            "distance_restaurant": "$location.recreation.distance_restaurant_m",
            "distance_children": "$location.social_facilities.distance_for_children_m",
            "distance_seniors": "$location.social_facilities.distance_for_senior_m",
            "distance_shelter": "$location.social_facilities.distance_shelter_m",
            "distance_hotel": "$location.tourism.distance_hotel_m",
            "distance_museum": "$location.tourism.distance_museum_m",
        }
    }

    stage_project_market_spec = {"$project": {"weather, economics..."}}

    # Fill the pipeline(s) with the necessary stages
    pipeline = [
        stage_projection_object_spec,
        # stage_sort_by_price,
    ]

    # Aggregation
    cursor = MONGO.collection.aggregate(pipeline)

    # Daten der Aggregation in Pandas Dataframe ziehen
    df_raw = pd.DataFrame.from_records(list(cursor))
    df_raw.to_csv("immonet_data_objectspec/raw_data.csv", index=False)
