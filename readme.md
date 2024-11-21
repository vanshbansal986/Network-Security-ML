# Network Security Data - End-to-End Machine Learning Project

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Components](#pipeline-components)
  - [Data Ingestion](#data-ingestion)
  - [Data Validation](#data-validation)
  - [Data Transformation](#data-transformation)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Model Pushing](#model-pushing)
- [MLFlow & Dagshub](#mlflow--dagshub)
- [FastAPI Interface](#fastapi-interface)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project is an end-to-end Machine Learning pipeline designed for Network Security Data analysis. The pipeline involves data ingestion from a MongoDB database, data validation, transformation, model training, evaluation, and model deployment. The objective is to predict network security events using a trained machine learning model.

Key components used in this project:
- **Database**: MongoDB
- **Experiment Tracking**: MLFlow
- **Remote Repository**: Dagshub
- **Web Interface**: FastAPI

## Project Structure
Here is a high-level view of the project structure:

![Project Structure](https://github.com/vanshbansal986/Network-Security-ML/blob/main/images/project_structure.png)

```bash
📂 network-security-ml-project
├── 📂 Artifacts                       # Directory for artifacts generated during training runs
│   ├── 📂 11_20_2024_00_36_05         # Artifacts with timestamp (YYYY_MM_DD_HH_MM_SS)
│   │   ├── 📂 data_ingestion          # This directory contains the ingested data from MongoDB
│   │   │   ├── 📂 feature_store
│   │   │   │   └── phisingData.csv    # Ingested data full 
│   │   │   └── 📂 ingested
│   │   │       ├── test.csv          # Ingested test data
│   │   │       └── train.csv         # Ingested train data
│   │   ├── 📂 data_transformation     # This directory contains the transformed ingested data in numpy array format and preprocessor object.
│   │   │   ├── 📂 transformed
│   │   │   │   ├── test.npy          # Tranformed test data
│   │   │   │   └── train.npy         # Tranformed train data
│   │   │   └── 📂 transformed_object
│   │   │       └── preprocessing.pkl    # Preprocessing pipeline object
│   │   ├── 📂 data_validation        # This directory contains the validated data and validation reports.
│   │   │   ├── 📂 drift_report    
│   │   │   │   └── report.yaml      # Drift report of ingested data compared to previous data
│   │   │   └── 📂 validated          # Folder containing validated data
│   │   │       ├── test.csv          # Validated test data
│   │   │       └── train.csv          # Validated train data
│   │   └── 📂 model_trainer          # Directory containing the trained Machine Learning model
│   │       └── 📂 trained_model
│   │           └── model.pkl        # Trained Machine Learning Model
│   ├── 📂 11_21_2024_11_54_42         # Another timestamped directory for artifacts
│   │   └── (similar structure as above)
│   ├── 📂 11_21_2024_12_20_13
│   │   └── (similar structure as above)
│   ├── 📂 11_21_2024_13_54_19
│   │   └── (similar structure as above)
│   └── 📂 11_21_2024_13_59_57
│       └── (similar structure as above)
├── 📜 Dockerfile                      # Docker configuration file for containerizing the application
├── 📂 NetworkSecurity.egg-info        # Metadata directory for the Python package
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   ├── requires.txt
│   └── top_level.txt
├── 📜 app.py                          # Script for deploying a web application (using FastAPI)
├── 📂 final_model                     # Directory for storing the final trained model and preprocessing object.
│   ├── model.pkl
│   └── preprocessor.pkl
├── 📂 logs                            # Logs generated during different runs for monitoring
├── 📜 main.py                         # Main entry point to run the pipeline or trigger components
├── 📂 network_data                    # Folder to store network-related datasets
│   └── phisingData.csv
├── 📂 network_security                # Main package for the project containing source code
│   ├── 📜 __init__.py                 # Logging configuration and utilities
│   ├── 📂 cloud                       # Code related to cloud operations (Not yet implemented)
│   │   └── 📜 __init__.py
│   ├── 📂 components                  # Code for each component of the pipeline
│   │   ├── 📜 __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   └── model_trainer.py
│   ├── 📂 constants                   # Constants used throughout the project
│   │   ├── 📜 __init__.py
│   │   └── 📂 training_pipeline
│   │       └── 📜 __init__.py
│   ├── 📂 entity                      # Data entities and configuration entities for the pipeline
│   │   ├── 📜 __init__.py
│   │   ├── artifact_entity.py
│   │   └── config_entity.py
│   ├── 📂 exception                   # Custom exception handling for the project
│   │   ├── 📜 __init__.py
│   │   └── exception.py
│   ├── 📂 logging                     
│   │   └── logger.py
│   ├── 📂 pipeline                    # Pipeline scripts for batch prediction and training
│   │   ├── 📜 __init__.py
│   │   ├── batch_prediction.py
│   │   └── training_pipeline.py
│   └── 📂 utils                       # Utility functions for common tasks
│       ├── 📜 __init__.py
│       ├── common.py
│       └── 📂 model                   # Model-related utilities (e.g., saving, loading)
│           ├── 📜 __init__.py
│           └── estimator.py
├── 📂 notebooks                       # Jupyter notebooks for EDA, model experiments, etc.
├── 📂 prediction_output               # Outputs from model predictions.
│   └── output.csv
├── 📜 push_data.py                    # Script to push data to a database or storage system
├── 📜 readme.md                       # Project README file
├── 📜 requirements.txt                # List of dependencies for the project
├── 📂 research                        # Research and experimentation notebooks
│   └── data_ingestion.ipynb
├── 📜 schema.yaml                     # Schema definition for validation of datasets
├── 📜 setup.py                        # Script to install the package
├── 📂 templates                       # HTML templates for web app(FastAPI)
│   └── table.html
├── 📜 test_mongodb.py                 # Script for testing MongoDB connections
└── 📂 valida_data                     # Folder for testing project working on FastAPI web app.
    └── test.csv
