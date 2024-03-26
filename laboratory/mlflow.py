# file: mlflow_utils.py

import os
import mlflow
from typing import Any
from typing import Optional
import pandas as pd


def set_mlflow_tracking_uri_from_env(env_vars):
    MLFLOW_TRACKING_URI = env_vars.get('MLFLOW_TRACKING_URI')
    print("MLFLOW_TRACKING_URI:", MLFLOW_TRACKING_URI) 
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print("mlflow tracking URI has been set to ", MLFLOW_TRACKING_URI)
    else:
        print("MLFLOW_TRACKING_URI not found in environment variables")



def create_mlflow_experiment(
    experiment_name: str, tags: dict[str, Any]
) -> str:
    """"""
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, tags=tags
        )
    except:
        print(f"Experiment {experiment_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name=experiment_name)
    return experiment_id


def get_mlflow_experiment(
    experiment_id: Optional[str] = None, experiment_name: Optional[str] = None
) -> mlflow.entities.Experiment:
    """Retrieves an MLflow experiment by its ID or name."""
    if experiment_id is not None:
        return mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment
        else:
            raise ValueError(f"Experiment named '{experiment_name}' does not exist.")
    else:
        raise ValueError("Either experiment_id or experiment_name must be provided.")
    
    
    
## From mlflow docs
def get_or_create_experiment(experiment_name):
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)  # Add this line to set the experiment
    return experiment_id