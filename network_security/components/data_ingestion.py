import pymongo.mongo_client
from network_security.exception.exception import NetworkSecurityException
from network_security import logger
from network_security.entity.config_entity import DataIngestionConfig
from network_security.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import pymongo
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self , config: DataIngestionConfig):
        try:
            self.config = config

        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def export_collection_as_df(self):
        try:
            
            db_name = self.config.database_name
            collection_name = self.config.collection_name

            print(f"Database Name: {db_name}, Collection Name: {collection_name}")
            
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[db_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            
            print(df.head())  # Debug: Check if the DataFrame has any data


            if "_id" in df.columns.to_list():
                df.drop(columns=["_id"] , axis=1 , inplace=True)

            df.replace({"na" : np.nan} , inplace = True)
            return df

        except Exception as e:
            raise NetworkSecurityException(e,sys)


    def export_data_to_feature_store(self , df:pd.DataFrame):
        try:
            feature_store_file_path = self.config.feature_store_file_path

            # creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path , exist_ok=True)
            
            df.to_csv(feature_store_file_path , index=False , header=True)

        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.config.train_test_split_ratio
            )
            logger.info("Performed train test split on the dataframe")

            logger.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            
            dir_path = os.path.dirname(self.config.training_file_path)
            
            os.makedirs(dir_path, exist_ok=True)
            
            logger.info(f"Exporting train and test file path.")
            
            train_set.to_csv(
                self.config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.config.testing_file_path, index=False, header=True
            )
            logger.info(f"Exported train and test file path.")

            
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def inititiate_data_ingestion(self):
        try:
            df = self.export_collection_as_df()
            self.export_data_to_feature_store(df)
            self.split_data_as_train_test(df)

            dataingestionartifact = DataIngestionArtifact(
                trained_file_path=self.config.training_file_path, 
                test_file_path=self.config.testing_file_path
                )
            
            return dataingestionartifact


        except Exception as e:
            raise NetworkSecurityException(e,sys)

