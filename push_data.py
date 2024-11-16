import os
import sys
import json
import certifi
import pymongo
import pandas as pd
import numpy as np
from network_security import logger
from network_security.exception.exception import NetworkSecurityException

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

ca = certifi.where()

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def csv_to_json_convertor(self , file_path):
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True , inplace=True)

            # Converting the df into a list of json values to store in mongodb
            #records = list(json.load(df.T.to_json()).values())
            
            records = df.to_dict(orient='records')


            return records
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def insert_data_to_mongodb(self , records , database , collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]

            self.collection.insert_many(self.records)

            return len(self.records)
        except Exception as e:
            raise NetworkSecurityException(e,sys)


if __name__ == "__main__":
    FILE_PATH = "network_data/phisingData.csv"
    DATABASE = "VanshAI"
    collection = "NetworkData"

    network_obj = NetworkDataExtract()
    
    records = network_obj.csv_to_json_convertor(FILE_PATH)

    no_of_records = network_obj.insert_data_to_mongodb(records , DATABASE , collection)

    print(no_of_records)



