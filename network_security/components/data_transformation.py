import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from network_security import logger
from network_security.utils.common import save_numpy_array_data , save_object
from network_security.entity.config_entity import DataTransformationConfig
from network_security.exception.exception import NetworkSecurityException
from network_security.constants.training_pipeline import TARGET_COLUMN
from network_security.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from network_security.entity.artifact_entity import DataTransformationArtifact , DataValidationArtifact


class DataTransformation:
    def __init__(self,
                data_validation_artifact:DataValidationArtifact,
                data_transformation_config: DataTransformationConfig
                ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    

    def get_data_transformer_object(cls)->Pipeline:
        """
        It initialises a KNNImputer object with the parameters specified in the training_pipeline.py file
        and returns a Pipeline object with the KNNImputer object as the first step.

        Args:
          cls: DataTransformation

        Returns:
          A Pipeline object
        """
        logger.info(
            "Entered get_data_trnasformer_object method of Trnasformation class"
        )
        try:
           imputer:KNNImputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
           logger.info(
                f"Initialise KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )
           processor:Pipeline=Pipeline([("imputer",imputer)])
           return processor
        except Exception as e:
            raise NetworkSecurityException(e,sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Data Transformation started")
            
            # Reading train and test data
            df_train = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            df_test = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            
            # Getting only input features
            input_train_df = df_train.drop(columns=[TARGET_COLUMN] , axis=1)
            input_test_df = df_test.drop(columns=[TARGET_COLUMN] , axis=1)

            # Replacing -1 with 0 in target column
            target_train_df = df_train[TARGET_COLUMN].replace(-1,0)
            target_test_df = df_test[TARGET_COLUMN].replace(-1,0)

            # Getting the pre-processor pipeline/object
            preprocessor=self.get_data_transformer_object()
            preprocessor_object=preprocessor.fit(input_train_df)

            # Transforming the data
            transformed_input_train = preprocessor_object.transform(input_train_df)
            transformed_input_test = preprocessor_object.transform(input_test_df)

            # Concatinating the input and target numpy arrays
            train_arr = np.c_[transformed_input_train, np.array(target_train_df) ]
            test_arr = np.c_[ transformed_input_test, np.array(target_test_df) ]

            
            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)

            save_object( "final_model/preprocessor.pkl", preprocessor_object,)


            #preparing artifacts

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)