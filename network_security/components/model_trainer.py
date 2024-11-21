from network_security import logger
from network_security.entity.config_entity import ModelTrainerConfig
from network_security.entity.config_entity import DataTransformationConfig
from network_security.entity.artifact_entity import DataTransformationArtifact , ModelTrainerArtifact , ClassificationMetricArtifact
from network_security.constants.training_pipeline import SCHEMA_FILE_PATH
from network_security.exception.exception import NetworkSecurityException
from network_security.utils.common import save_object , load_object , load_numpy_array_data , evaluate_models
from network_security.utils.common import get_classification_score
from network_security.utils.model.estimator import NetworkModel
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier , GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import sys
import os
import mlflow
from urllib.parse import urlparse

import dagshub
dagshub.init(repo_owner='vanshbansal986', repo_name='Network-Security-ML', mlflow=True)


class ModelTrainer:
    def __init__(self,
                model_trainer_config: ModelTrainerConfig,
                data_transformation_artifact: DataTransformationArtifact
                ) -> None:
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def track_mlflow(self , best_model , classification_metric):
        with mlflow.start_run():
            
            # mlflow.set_registry_uri("https://dagshub.com/krishnaik06/networksecurity.mlflow")
            # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            f1_score = classification_metric.f1_score
            precison = classification_metric.precision_score
            recall_score = classification_metric.recall_score

            mlflow.log_metric("f1 score" , f1_score)
            mlflow.log_metric("precison" , precison)
            mlflow.log_metric("recall_score" , recall_score)
            mlflow.sklearn.log_model(best_model , "best model")

            # Model registry does not work with file store
            # if tracking_url_type_store != "file":

            #     # Register the model
            #     # There are other ways to use the Model Registry, which depends on the use case,
            #     # please refer to the doc for more information:
            #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            #     mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model)
            # else:
            #     mlflow.sklearn.log_model(best_model, "model")

    def train_model(self , x_train,y_train,x_test,y_test):
        models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                # 'learning_rate':[.1,.01,.05,.001],
                # 'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }

        model_report:dict=evaluate_models(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                        models=models,param=params)
        
         ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

          ## To get best model name from dict
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        
        # Predicting y_train_pred using best model
        y_train_pred=best_model.predict(x_train)
        classification_train_metric = get_classification_score(y_true=y_train,y_pred=y_train_pred)

        # Predicting y_test_pred using best model
        y_test_pred=best_model.predict(x_test)
        classification_test_metric = get_classification_score(y_true=y_test,y_pred=y_test_pred)

        # Tracking using MLFlow
        self.track_mlflow(best_model , classification_train_metric)
        self.track_mlflow(best_model , classification_test_metric)

        preprocessor = load_object(file_path = self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Network_Model = NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj = Network_Model)
        #model pusher
        save_object("final_model/model.pkl",best_model)


         ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(
                            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                            train_metric_artifact=classification_train_metric,
                            test_metric_artifact=classification_test_metric
                            )
        
        logger.info(f"Model trainer artifact: {model_trainer_artifact}")
        
        return model_trainer_artifact
        
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:

            # Loading the train and test data in numpy array format
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(x_train,y_train,x_test,y_test)
            
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)