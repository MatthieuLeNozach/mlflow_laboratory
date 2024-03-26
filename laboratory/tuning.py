# file: tuning_utils.py

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
from typing import Dict, List, Iterable


def get_classification_metrics(
    y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame, prefix: str
) -> Dict[str, float]:
    """Gets classification metrics: accuracy, precision, recall, f1, and ROC AUC."""
    metrics = {
        f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
        f'{prefix}_precision': precision_score(y_true, y_pred),
        f'{prefix}_recall': recall_score(y_true, y_pred),
        f'{prefix}_f1_score': f1_score(y_true, y_pred),
        f'{prefix}_roc_auc': roc_auc_score(y_true, y_pred_proba[:, 1]),  # Assuming y_pred_proba is a 2D array
    }
    return metrics


def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.
    """
    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")



def objective_function(trial, X_train, X_test, y_train, y_test, pipeline, param_space):
    """Objective function for hyperparameter optimization."""
    params = {}
    for k, v in param_space.items():
        if isinstance(v, tuple):
            if len(v) == 2:
                params[k] = trial.suggest_int(k, v[0], v[1])
            elif len(v) == 3 and v[2] == 'log':
                params[k] = trial.suggest_loguniform(k, v[0], v[1])
        elif isinstance(v, list):
            params[k] = trial.suggest_categorical(k, v)
        else:
            params[k] = v

    pipeline.set_params(**params)

    with mlflow.start_run(nested=True):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)
        metrics = get_classification_metrics(y_test, y_pred, y_pred_proba, prefix='test')
        
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        return metrics['test_f1_score']


def map_best_params(best_params, param_space):
    mapped_params = {}
    for k, v in best_params.items():
        if k in param_space:
            if isinstance(param_space[k], tuple):
                if len(param_space[k]) == 2:
                    mapped_params[k] = int(v)  # Convert to int for integer parameters
                else:
                    mapped_params[k] = v
            elif isinstance(param_space[k], list):
                mapped_params[k] = param_space[k][int(v)]  # Convert index to int and retrieve value
            else:
                mapped_params[k] = v
        else:
            mapped_params[k] = v
    return mapped_params