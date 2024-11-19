from network_security.components.data_ingestion import DataIngestion
from network_security.components.data_validation import DataValidation
from network_security.components.data_transformation import DataTransformation
from network_security.components.model_trainer import ModelTrainer
from network_security.entity.config_entity import DataIngestionConfig , DataValidationConfig , DataTransformationConfig , ModelTrainerConfig
from network_security.entity.config_entity import TrainingPipelineConfig
from network_security import logger
from network_security.exception.exception import NetworkSecurityException

import sys

if __name__=='__main__':
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config)
        logger.info("Initiate the data ingestion")
        data_ingestion_artifact = data_ingestion.inititiate_data_ingestion()
        logger.info("Data ingestion completed!!!")

        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact , data_validation_config)
        logger.info("Initiate the data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logger.info("Data validation completed!!!")

        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact , data_transformation_config)
        logger.info("Initiate the data transformation...")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logger.info("Data transformation completed!!!")

        logger.info("Model Training started")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config , data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logger.info("Model Training finished!!!")


    except Exception as e:
        raise NetworkSecurityException(e,sys)


