from network_security.components.data_ingestion import DataIngestion
from network_security.entity.config_entity import DataIngestionConfig
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
        dataingestionartifact=data_ingestion.inititiate_data_ingestion()
    except Exception as e:
        raise NetworkSecurityException(e,sys)
