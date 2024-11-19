from network_security import logger
from network_security.entity.config_entity import DataIngestionConfig
from network_security.entity.config_entity import DataValidationConfig
from network_security.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact
from network_security.constants.training_pipeline import SCHEMA_FILE_PATH
from network_security.exception.exception import NetworkSecurityException
from network_security.utils.common import read_yaml , write_yaml
from scipy.stats import ks_2samp
import pandas as pd
import sys
import os



class DataValidation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig
                ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml(SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns=len(self.schema_config['columns'])
            logger.info(f"Required number of columns:{number_of_columns}")
            logger.info(f"Data frame has columns:{len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    
                    }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml(file_path=drift_report_file_path,content=report)

            return status

        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        
        try:
            # Getting the current train and test file path
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Reading the data from train and test
            train_df = pd.read_csv(train_file_path)
            logger.info("Train file path read successfully!!!")
            test_df = pd.read_csv(test_file_path)
            logger.info("Test file path read successfully!!!")

            ## validate number of columns

            status = self.validate_number_of_columns(dataframe=train_df)
            if not status:
                error_message=f"Train dataframe does not contain all columns.\n"
            
            status = self.validate_number_of_columns(dataframe=test_df)
            if not status:
                error_message=f"Test dataframe does not contain all columns.\n"   

            ## lets check datadrift
            status = self.detect_dataset_drift(base_df=train_df,current_df=test_df)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_df.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True

            )

            test_df.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path = train_file_path,
                valid_test_file_path = test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact

        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        