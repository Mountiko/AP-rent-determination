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
from pandas import DataFrame


NAME = "Demo für Mohammad"

logging.basicConfig(
    level=logging.INFO,
    format=f"[%(asctime)s] {NAME} - %(levelname)s: %(message)s",
    filename="logging.log",
    filemode="w",
)

MONGO = MongoClient()


if __name__ == "__main__":

    MONGO.set_collection("students", "wohnung_kaufen")

    # Sort stages
    stage_sort_by_obseravtionDate = {"$sort": {"observationDate": pymongo.DESCENDING}}

    # Project stages

    stage_projection_all = {
        "$project": {
            "observationDate": "$observationDate",
            "state": "$state",
            "city": "$city",
            "AP_community": "$AP_community",
            "community_id": "$community_id",
            "postcode": "$postcode",
            "livingSpace": "$livingArea",
            "roomCount": "$rooms",
            "floor": "$level",
            "constructionYear": "$constructionYear",
            "apartmentType": "$apartmentType",
            "condition": "$condition",
            "price": "$price",
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
            "nearest_kindergarden": {
                "$arrayElemAt": ["$location.education.distance_kindergarden_m", 0]
            },
            "second_kindergarden": {
                "$arrayElemAt": ["$location.education.distance_kindergarden_m", 1]
            },
            "third_kindergarden": {
                "$arrayElemAt": ["$location.education.distance_kindergarden_m", 2]
            },
            "nearest_school": {
                "$arrayElemAt": ["$location.education.distance_school_m", 0]
            },
            "second_school": {
                "$arrayElemAt": ["$location.education.distance_school_m", 1]
            },
            "third_school": {
                "$arrayElemAt": ["$location.education.distance_school_m", 2]
            },
            "nearest_university": {
                "$arrayElemAt": ["$location.education.distance_university_m", 0]
            },
            "second_university": {
                "$arrayElemAt": ["$location.education.distance_university_m", 1]
            },
            "third_university": {
                "$arrayElemAt": ["$location.education.distance_university_m", 2]
            },
            "nearest_ATM": {
                "$arrayElemAt": ["$location.general_needs.distance_ATM_m", 0]
            },
            "second_ATM": {
                "$arrayElemAt": ["$location.general_needs.distance_ATM_m", 1]
            },
            "third_ATM": {
                "$arrayElemAt": ["$location.general_needs.distance_ATM_m", 2]
            },
            "nearest_bakery": {
                "$arrayElemAt": ["$location.general_needs.distance_bakery_m", 0]
            },
            "second_bakery": {
                "$arrayElemAt": ["$location.general_needs.distance_bakery_m", 1]
            },
            "third_bakery": {
                "$arrayElemAt": ["$location.general_needs.distance_bakery_m", 2]
            },
            "nearest_DIY_store": {
                "$arrayElemAt": [
                    "$location.general_needs.distance_doityourself_store_m",
                    0,
                ]
            },
            "second_DIY_store": {
                "$arrayElemAt": [
                    "$location.general_needs.distance_doityourself_store_m",
                    1,
                ]
            },
            "third_DIY_store": {
                "$arrayElemAt": [
                    "$location.general_needs.distance_doityourself_store_m",
                    2,
                ]
            },
            "nearest_hairdresser": {
                "$arrayElemAt": ["$location.general_needs.distance_hairdresser_m", 0]
            },
            "second_hairdresser": {
                "$arrayElemAt": ["$location.general_needs.distance_hairdresser_m", 1]
            },
            "third_hairdresser": {
                "$arrayElemAt": ["$location.general_needs.distance_hairdresser_m", 2]
            },
            "nearest_supermarket": {
                "$arrayElemAt": ["$location.general_needs.distance_supermarket_m", 0]
            },
            "second_supermarket": {
                "$arrayElemAt": ["$location.general_needs.distance_supermarket_m", 1]
            },
            "third_supermarket": {
                "$arrayElemAt": ["$location.general_needs.distance_supermarket_m", 2]
            },
            "nearest_clinic": {
                "$arrayElemAt": ["$location.health.distance_clinic_m", 0]
            },
            "second_clinic": {
                "$arrayElemAt": ["$location.health.distance_clinic_m", 1]
            },
            "third_clinic": {"$arrayElemAt": ["$location.health.distance_clinic_m", 2]},
            "nearest_doctor": {
                "$arrayElemAt": ["$location.health.distance_doctor_m", 0]
            },
            "second_doctor": {
                "$arrayElemAt": ["$location.health.distance_doctor_m", 1]
            },
            "third_doctor": {"$arrayElemAt": ["$location.health.distance_doctor_m", 2]},
            "nearest_hospital": {
                "$arrayElemAt": ["$location.health.distance_hospital_m", 0]
            },
            "second_hospital": {
                "$arrayElemAt": ["$location.health.distance_hospital_m", 1]
            },
            "third_hospital": {
                "$arrayElemAt": ["$location.health.distance_hospital_m", 2]
            },
            "nearest_pharmacy": {
                "$arrayElemAt": ["$location.health.distance_pharmacy_m", 0]
            },
            "second_pharmacy": {
                "$arrayElemAt": ["$location.health.distance_pharmacy_m", 1]
            },
            "third_pharmacy": {
                "$arrayElemAt": ["$location.health.distance_pharmacy_m", 2]
            },
            "nearest_airport": {
                "$arrayElemAt": ["$location.infrastructure.distance_airport_m", 0]
            },
            "second_airport": {
                "$arrayElemAt": ["$location.infrastructure.distance_airport_m", 1]
            },
            "third_airport": {
                "$arrayElemAt": ["$location.infrastructure.distance_airport_m", 2]
            },
            "nearest_bus_stop": {
                "$arrayElemAt": ["$location.infrastructure.distance_bus_stop_m", 0]
            },
            "second_bus_stop": {
                "$arrayElemAt": ["$location.infrastructure.distance_bus_stop_m", 1]
            },
            "third_bus_stop": {
                "$arrayElemAt": ["$location.infrastructure.distance_bus_stop_m", 2]
            },
            "nearest_charging_station": {
                "$arrayElemAt": [
                    "$location.infrastructure.distance_charging_stations_m",
                    0,
                ]
            },
            "second_charging_station": {
                "$arrayElemAt": [
                    "$location.infrastructure.distance_charging_stations_m",
                    1,
                ]
            },
            "third_charging_station": {
                "$arrayElemAt": [
                    "$location.infrastructure.distance_charging_stations_m",
                    2,
                ]
            },
            "nearest_fuel": {
                "$arrayElemAt": ["$location.infrastructure.distance_fuel_m", 0]
            },
            "second_fuel": {
                "$arrayElemAt": ["$location.infrastructure.distance_fuel_m", 1]
            },
            "third_fuel": {
                "$arrayElemAt": ["$location.infrastructure.distance_fuel_m", 2]
            },
            "nearest_harbour": {
                "$arrayElemAt": ["$location.infrastructure.distance_harbours_m", 0]
            },
            "second_harbour": {
                "$arrayElemAt": ["$location.infrastructure.distance_harbours_m", 1]
            },
            "third_harbour": {
                "$arrayElemAt": ["$location.infrastructure.distance_harbours_m", 2]
            },
            "nearest_motorway_junction": {
                "$arrayElemAt": [
                    "$location.infrastructure.distance_motorway_junction_m",
                    0,
                ]
            },
            "second_motorway_junction": {
                "$arrayElemAt": [
                    "$location.infrastructure.distance_motorway_junction_m",
                    1,
                ]
            },
            "third_motorway_junction": {
                "$arrayElemAt": [
                    "$location.infrastructure.distance_motorway_junction_m",
                    2,
                ]
            },
            "nearest_recycling_center": {
                "$arrayElemAt": [
                    "$location.infrastructure.distance_recycling_center_m",
                    0,
                ]
            },
            "second_recycling_center": {
                "$arrayElemAt": [
                    "$location.infrastructure.distance_recycling_center_m",
                    1,
                ]
            },
            "third_recycling_center": {
                "$arrayElemAt": [
                    "$location.infrastructure.distance_recycling_center_m",
                    2,
                ]
            },
            "nearest_train_station": {
                "$arrayElemAt": ["$location.infrastructure.distance_train_station_m", 0]
            },
            "second_train_station": {
                "$arrayElemAt": ["$location.infrastructure.distance_train_station_m", 1]
            },
            "third_train_station": {
                "$arrayElemAt": ["$location.infrastructure.distance_train_station_m", 2]
            },
            "nearest_tram_station": {
                "$arrayElemAt": ["$location.infrastructure.distance_tram_stations_m", 0]
            },
            "second_tram_station": {
                "$arrayElemAt": ["$location.infrastructure.distance_tram_stations_m", 1]
            },
            "third_tram_station": {
                "$arrayElemAt": ["$location.infrastructure.distance_tram_stations_m", 2]
            },
            "nearest_bar": {"$arrayElemAt": ["$location.recreation.distance_bar_m", 0]},
            "second_bar": {"$arrayElemAt": ["$location.recreation.distance_bar_m", 1]},
            "third_bar": {"$arrayElemAt": ["$location.recreation.distance_bar_m", 2]},
            "nearest_beergarden": {
                "$arrayElemAt": ["$location.recreation.distance_beergarden_m", 0]
            },
            "second_beergarden": {
                "$arrayElemAt": ["$location.recreation.distance_beergarden_m", 1]
            },
            "third_beergarden": {
                "$arrayElemAt": ["$location.recreation.distance_beergarden_m", 2]
            },
            "nearest_nightclub": {
                "$arrayElemAt": ["$location.recreation.distance_nightclub_m", 0]
            },
            "second_nightclub": {
                "$arrayElemAt": ["$location.recreation.distance_nightclub_m", 1]
            },
            "third_nightclub": {
                "$arrayElemAt": ["$location.recreation.distance_nightclub_m", 2]
            },
            "nearest_restaurant": {
                "$arrayElemAt": ["$location.recreation.distance_restaurant_m", 0]
            },
            "second_restaurant": {
                "$arrayElemAt": ["$location.recreation.distance_restaurant_m", 1]
            },
            "third_restaurant": {
                "$arrayElemAt": ["$location.recreation.distance_restaurant_m", 2]
            },
            "nearest_children": {
                "$arrayElemAt": [
                    "$location.social_facilities.distance_for_children_m",
                    0,
                ]
            },
            "second_children": {
                "$arrayElemAt": [
                    "$location.social_facilities.distance_for_children_m",
                    1,
                ]
            },
            "third_children": {
                "$arrayElemAt": [
                    "$location.social_facilities.distance_for_children_m",
                    2,
                ]
            },
            "nearest_seniors": {
                "$arrayElemAt": ["$location.social_facilities.distance_for_senior_m", 0]
            },
            "second_seniors": {
                "$arrayElemAt": ["$location.social_facilities.distance_for_senior_m", 1]
            },
            "third_seniors": {
                "$arrayElemAt": ["$location.social_facilities.distance_for_senior_m", 2]
            },
            "nearest_shelter": {
                "$arrayElemAt": ["$location.social_facilities.distance_shelter_m", 0]
            },
            "second_shelter": {
                "$arrayElemAt": ["$location.social_facilities.distance_shelter_m", 1]
            },
            "third_shelter": {
                "$arrayElemAt": ["$location.social_facilities.distance_shelter_m", 2]
            },
            "nearest_hotel": {
                "$arrayElemAt": ["$location.tourism.distance_hotel_m", 0]
            },
            "second_hotel": {"$arrayElemAt": ["$location.tourism.distance_hotel_m", 1]},
            "third_hotel": {"$arrayElemAt": ["$location.tourism.distance_hotel_m", 2]},
            "nearest_museum": {
                "$arrayElemAt": ["$location.tourism.distance_museum_m", 0]
            },
            "second_museum": {
                "$arrayElemAt": ["$location.tourism.distance_museum_m", 1]
            },
            "third_museum": {
                "$arrayElemAt": ["$location.tourism.distance_museum_m", 2]
            },
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
            "base_rent": "$rent.rentBase",
            "additionalCost": "$rent.additionalCost",
            "heatingCost": "$rent.heatingCost",
            "total_rent": "$rent.totalRent",
            "total_rent_includes_heating": "$rent.totalRentIncludesHeating",
            "deposit": "$rent.deposit",
            "parkingCount": "$rent.parking.parkingCount",
            "rentPerParking": "$rent.parking.rentPerParking",
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
            "isCommonResidentialProperty": "$isCommonResidentialProperty",
            "approved": "$approved",
            "propertyTransferTax": "$cost.propertyTransferTax",
        }
    }

    stage_project_location_spec = {
        "$project": {
            "longitude": "$coords.lng",
            "latitude": "$coords.lat",
            "coord_confidence": "$coords.confidence",
            "nearest_kindergarden": {
                "$arrayElemAt": ["$location.education.distance_kindergarden_m", 0]
            },
            "count_kindergarden_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.education.distance_kindergarden_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_kindergarden_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.education.distance_kindergarden_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_kindergarden_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.education.distance_kindergarden_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_kindergarden_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.education.distance_kindergarden_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_school": {
                "$arrayElemAt": ["$location.education.distance_school_m", 0]
            },
            "count_school_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.education.distance_school_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_school_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.education.distance_school_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_school_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.education.distance_school_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_school_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.education.distance_school_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_university": {
                "$arrayElemAt": ["$location.education.distance_university_m", 0]
            },
            "count_university_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.education.distance_university_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_university_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.education.distance_university_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_university_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.education.distance_university_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_university_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.education.distance_university_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_ATM": {
                "$arrayElemAt": ["$location.general_needs.distance_ATM_m", 0]
            },
            "count_ATM_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_ATM_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_ATM_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_ATM_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_ATM_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_ATM_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_ATM_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_ATM_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_bakery": {
                "$arrayElemAt": ["$location.general_needs.distance_bakery_m", 0]
            },
            "count_bakery_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_bakery_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_bakery_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_bakery_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_bakery_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_bakery_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_bakery_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_bakery_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_DIY_store": {
                "$arrayElemAt": [
                    "$location.general_needs.distance_doityourself_store_m",
                    0,
                ]
            },
            "count_DIY_store_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_doityourself_store_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_DIY_store_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_doityourself_store_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_DIY_store_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_doityourself_store_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_DIY_store_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_doityourself_store_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_hairdresser": {
                "$arrayElemAt": ["$location.general_needs.distance_hairdresser_m", 0]
            },
            "count_hairdresser_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_hairdresser_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_hairdresser_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_hairdresser_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_hairdresser_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_hairdresser_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_hairdresser_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_hairdresser_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_supermarket": {
                "$arrayElemAt": ["$location.general_needs.distance_supermarket_m", 0]
            },
            "count_supermarket_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_supermarket_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_supermarket_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_supermarket_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_supermarket_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_supermarket_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_supermarket_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.general_needs.distance_supermarket_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_clinic": {
                "$arrayElemAt": ["$location.health.distance_clinic_m", 0]
            },
            "count_clinic_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_clinic_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_clinic_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_clinic_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_clinic_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_clinic_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_clinic_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_clinic_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_doctor": {
                "$arrayElemAt": ["$location.health.distance_doctor_m", 0]
            },
            "count_doctor_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_doctor_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_doctor_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_doctor_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_doctor_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_doctor_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_doctor_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_doctor_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_hospital": {
                "$arrayElemAt": ["$location.health.distance_hospital_m", 0]
            },
            "count_hospital_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_hospital_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_hospital_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_hospital_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_hospital_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_hospital_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_hospital_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_hospital_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_pharmacy": {
                "$arrayElemAt": ["$location.health.distance_pharmacy_m", 0]
            },
            "count_pharmacy_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_pharmacy_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_pharmacy_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_pharmacy_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_pharmacy_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_pharmacy_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_pharmacy_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.health.distance_pharmacy_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_airport": {
                "$arrayElemAt": ["$location.infrastructure.distance_airport_m", 0]
            },
            "count_airport_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_airport_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_airport_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_airport_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_airport_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_airport_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_airport_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_airport_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_bus_stop": {
                "$arrayElemAt": ["$location.infrastructure.distance_bus_stop_m", 0]
            },
            "count_bus_stop_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_bus_stop_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_bus_stop_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_bus_stop_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_bus_stop_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_bus_stop_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_bus_stop_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_bus_stop_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_charging_station": {
                "$arrayElemAt": [
                    "$location.infrastructure.distance_charging_stations_m",
                    0,
                ]
            },
            "count_charging_station_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_charging_stations_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_charging_station_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_charging_stations_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_charging_station_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_charging_stations_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_charging_station_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_charging_stations_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_fuel": {
                "$arrayElemAt": ["$location.infrastructure.distance_fuel_m", 0]
            },
            "count_fuel_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_fuel_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_fuel_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_fuel_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_fuel_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_fuel_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_fuel_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_fuel_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_harbour": {
                "$arrayElemAt": ["$location.infrastructure.distance_harbours_m", 0]
            },
            "count_harbour_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_harbours_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_harbour_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_harbours_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_harbour_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_harbours_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_harbour_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_harbours_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_motorway_junction": {
                "$arrayElemAt": [
                    "$location.infrastructure.distance_motorway_junction_m",
                    0,
                ]
            },
            "count_motorway_junction_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_motorway_junction_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_motorway_junction_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_motorway_junction_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_motorway_junction_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_motorway_junction_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_motorway_junction_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_motorway_junction_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_recycling_center": {
                "$arrayElemAt": [
                    "$location.infrastructure.distance_recycling_center_m",
                    0,
                ]
            },
            "count_recycling_center_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_recycling_center_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_recycling_center_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_recycling_center_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_recycling_center_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_recycling_center_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_recycling_center_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_recycling_center_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_train_station": {
                "$arrayElemAt": ["$location.infrastructure.distance_train_station_m", 0]
            },
            "count_train_station_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_train_station_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_train_station_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_train_station_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_train_station_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_train_station_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_train_station_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_train_station_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_tram_station": {
                "$arrayElemAt": ["$location.infrastructure.distance_tram_stations_m", 0]
            },
            "count_tram_station_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_tram_stations_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_tram_station_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_tram_stations_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_tram_station_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_tram_stations_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_tram_station_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.infrastructure.distance_tram_stations_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_bar": {"$arrayElemAt": ["$location.recreation.distance_bar_m", 0]},
            "count_bar_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_bar_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_bar_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_bar_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_bar_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_bar_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_bar_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_bar_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_beergarden": {
                "$arrayElemAt": ["$location.recreation.distance_beergarden_m", 0]
            },
            "count_beergarden_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_beergarden_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_beergarden_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_beergarden_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_beergarden_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_beergarden_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_beergarden_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_beergarden_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_nightclub": {
                "$arrayElemAt": ["$location.recreation.distance_nightclub_m", 0]
            },
            "count_nightclub_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_nightclub_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_nightclub_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_nightclub_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_nightclub_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_nightclub_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_nightclub_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_nightclub_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_restaurant": {
                "$arrayElemAt": ["$location.recreation.distance_restaurant_m", 0]
            },
            "count_restaurant_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_restaurant_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_restaurant_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_restaurant_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_restaurant_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_restaurant_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_restaurant_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.recreation.distance_restaurant_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_children": {
                "$arrayElemAt": [
                    "$location.social_facilities.distance_for_children_m",
                    0,
                ]
            },
            "count_children_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.social_facilities.distance_for_children_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_children_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.social_facilities.distance_for_children_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_children_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.social_facilities.distance_for_children_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_children_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.social_facilities.distance_for_children_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_seniors": {
                "$arrayElemAt": ["$location.social_facilities.distance_for_senior_m", 0]
            },
            "count_seniors_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.social_facilities.distance_for_senior_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_seniors_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.social_facilities.distance_for_senior_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_seniors_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.social_facilities.distance_for_senior_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_seniors_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.social_facilities.distance_for_senior_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_shelter": {
                "$arrayElemAt": ["$location.social_facilities.distance_shelter_m", 0]
            },
            "count_shelter_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.social_facilities.distance_shelter_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_shelter_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.social_facilities.distance_shelter_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_shelter_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.social_facilities.distance_shelter_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_shelter_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.social_facilities.distance_shelter_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_hotel": {
                "$arrayElemAt": ["$location.tourism.distance_hotel_m", 0]
            },
            "count_hotel_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.tourism.distance_hotel_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_hotel_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.tourism.distance_hotel_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_hotel_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.tourism.distance_hotel_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_hotel_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.tourism.distance_hotel_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
            "nearest_museum": {
                "$arrayElemAt": ["$location.tourism.distance_museum_m", 0]
            },
            "count_museum_1km": {
                "$size": {
                    "$filter": {
                        "input": "$location.tourism.distance_museum_m",
                        "cond": {"$lt": ["$$this", 1000]},
                    }
                }
            },
            "count_museum_5km": {
                "$size": {
                    "$filter": {
                        "input": "$location.tourism.distance_museum_m",
                        "cond": {"$lt": ["$$this", 5000]},
                    }
                }
            },
            "count_museum_10km": {
                "$size": {
                    "$filter": {
                        "input": "$location.tourism.distance_museum_m",
                        "cond": {"$lt": ["$$this", 10000]},
                    }
                }
            },
            "count_museum_50km": {
                "$size": {
                    "$filter": {
                        "input": "$location.tourism.distance_museum_m",
                        "cond": {"$lt": ["$$this", 50000]},
                    }
                }
            },
        }
    }

    stage_project_market_spec = {"$project": {"weather, economics..."}}

    # Fill the pipeline(s) with the necessary stages
    pipeline = [
        stage_projection_all,
        # stage_sort_by_price,
    ]

    # Aggregation
    cursor = MONGO.collection.aggregate(pipeline)

    # Use this cursor to get all columns from all documents in the collection
    # cursor = MONGO.collection.find()

    # Daten der Aggregation in Pandas Dataframe ziehen
    df_raw = pd.DataFrame.from_records(list(cursor))
    df_raw.to_csv("wohnung_kaufen/raw_data.csv", index=False)
