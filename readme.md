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

![Project Structure](https://github.com/vanshbansal986/Network-Security-ML/blob/main/images2/project_structure.png)

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

```

# Pipeline Components

## Data Ingestion

The Data Ingestion phase is crucial for setting up the initial dataset required for the Machine Learning workflow. This step ingests data from a MongoDB database and structures it for further processing. Below is a breakdown of the Data Ingestion process:

![Data Ingestion](https://your-image-url-here)

#### Key Steps:

1. **Configuration:**
   - The **Data Ingestion Config** specifies various paths:
     - **Data Ingestion Directory**: Directory where the ingested data will be stored.
     - **Feature Store File Path**: Path for the feature store which holds the raw data.
     - **Training and Testing File Paths**: Paths where training and testing datasets will be saved.
     - **Collection Name**: Name of the MongoDB collection from which data will be ingested.
     - **Train-Test Split Ratio**: Defines the proportion of data allocated for training versus testing.
2. **Initiate Data Ingestion:**
   - The process begins by calling the data ingestion function, which connects to the specified MongoDB database.

3. **Export Data to Feature Store:**
   - The data is exported to the feature store as a CSV file. This is the raw data and serves as the baseline for all subsequent processes.
4. **Data Ingestion Artifact:**
   - Finally, a **Data Ingestion Artifact** is created to maintain metadata about the ingestion process, including timestamps and paths to the ingested files.

#### Output:
- The end result of the Data Ingestion process is the creation of:
  - **Feature Store** containing the raw dataset as CSV.
  - **Ingested Data** folders containing `train.csv` and `test.csv` files for subsequent processing.

This structured approach ensures that the data flow into the pipeline is efficient, clean, and well-documented, providing a strong foundation for further stages of the data processing pipeline.

## Data Validation

The Data Validation phase ensures that the ingested data meets the necessary quality standards before proceeding to the next steps in the Machine Learning pipeline. The following diagram outlines the key aspects of this process:

![Data Validation](https://github.com/vanshbansal986/Network-Security-ML/blob/main/images2/data_validation.png)

### Overview of the Process:

1. **Configuration**:
   - The process starts with a **Data Validation Config** that defines directories for valid and invalid data, as well as paths for drift reports.

2. **Initiate Validation**:
   - Data validation is initiated, starting with reading the ingested CSV files (`train.csv` and `test.csv`).

3. **Column Validation**:
   - The number of columns and their data types is validated against the predefined schema. This includes checks to ensure the correct columns are present and that numerical columns exist.

4. **Validation Status**:
   - The validation status is recorded, indicating whether any columns are missing or if there are issues with the data types.

5. **Dataset Drift Detection**:
   - If the initial validation passes, the process checks for dataset drift, which helps ensure that the data remains consistent with the training parameters.

6. **Data Validation Artifact**:
   - Finally, a **Data Validation Artifact** is created, summarizing the validation status and producing a drift report in JSON format.


## Data Transformation

The Data Transformation phase prepares the validated data for modeling by handling missing values and scaling features. The following diagram illustrates the key steps in this process:

![Data Transformation](https://github.com/vanshbansal986/Network-Security-ML/blob/main/images2/data_transformation.png)

### Overview of the Process:

1. **Configuration**:
   - The process begins with a **Data Transformation Config**, which sets the parameters for the transformation pipeline.

2. **Initiate Data Transformation**:
   - The data transformation process is initiated, starting with the reading of the validated CSV files (`train.csv` and `test.csv`).

3. **Feature Handling**:
   - Missing values in the dataset are handled using techniques such as Robust Scaling and KNN Imputer. The target column is dropped from the training data to prepare for transformation.

4. **Data Preparation**:
   - The training and testing data are converted into feature arrays, preparing for input into the ML model. Techniques like SMOTE are applied to address imbalances in the data.

5. **Concatenation and Finalization**:
   - The final transformed training and testing arrays are created and concatenated accordingly.

6. **Artifact Creation**:
   - The transformation includes saving a preprocessor object (`preprocessing.pkl`) and the resulting numpy arrays (`train.npy` and `test.npy`) as artifacts for future use.


## Model Training

The Model Training phase is vital for building and optimizing a machine learning model based on the transformed data. The following diagram illustrates the core steps involved in this process:

![Model Training](https://github.com/vanshbansal986/Network-Security-ML/blob/main/images2/model_trainer.png)

### Overview of the Process:

1. **Configuration**:
   - The process begins with a **Model Trainer Config** that defines parameters such as model file paths and expected accuracy.

2. **Initiate Model Training**:
   - Model training is initiated by loading the transformed numpy arrays (`train.npy` and `test.npy`) for further processing.

3. **Data Preparation**:
   - The training and testing arrays are prepared by splitting the features and target values, ensuring the data is structured for model input.

4. **Model Selection**:
   - Multiple models are evaluated, including:
     - Random Forest
     - Decision Tree
     - Gradient Boosting
     - Logistic Regression
     - AdaBoost
   - GridSearchCV is utilized for hyperparameter tuning to find the best combination of parameters for each model.

5. **Model Evaluation**:
   - The best-performing model is selected based on evaluation metrics, including the best score compared to the expected accuracy.

6. **Artifact Creation**:
   - After training, the best model is saved as an object (`model.pkl`) along with associated metrics, which are logged for future reference.
