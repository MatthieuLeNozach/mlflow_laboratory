# mlflow_lab/script.py

def get_sklearn_binary_class():
    script_content = """
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import optuna
from functools import partial
import warnings
warnings.filterwarnings("ignore")
import mlflow

# Custom modules
import mlflow_lab.dataset as dataset
from mlflow_lab.mlflow import get_or_create_experiment
import mlflow_lab.pipeline as pipeline
import mlflow_lab.tuning as tuning
from mlflow_lab.hp_bank import RFC_SPACE

# Load environment variables
from dotenv import load_dotenv
load_dotenv('.env')
load_dotenv('.env.secret')

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set MinIO as the default artifact store
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MINIO_ENDPOINT_URL')
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('MINIO_ACCESS_KEY')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('MINIO_SECRET_KEY')

print(os.environ['MLFLOW_S3_ENDPOINT_URL'])
print(os.environ['AWS_ACCESS_KEY_ID'])
print(os.environ['AWS_SECRET_ACCESS_KEY'])

# Experiment setup
EXPERIMENT_NAME = 'optuna_experiment10'
RUN_NAME = 'hyperparameter_optimization'
DATASET_PATH = '../../0_DATASETS/creditcard.csv'
CLASSIFIER = RandomForestClassifier(bootstrap=False)
SPACE = RFC_SPACE
TARGET_NAME = 'Class'

#################### MAIN ####################

if __name__ == '__main__':
    df = pd.read_csv(DATASET_PATH)
    df_redux = dataset.data_split_redux(df, TARGET_NAME, zero_label_redux=0.995)

    X_train, X_test, y_train, y_test = train_test_split(
        df_redux.drop(columns=TARGET_NAME), df_redux[TARGET_NAME], test_size=0.2, random_state=42
    )

    num_features = X_train.select_dtypes([np.number]).columns.tolist()
    cat_features = X_train.columns.difference(num_features).tolist()

    pipeline = pipeline.get_binary_rfc_pipeline(num_features, cat_features)

    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):
        # Create an Optuna study
        study = optuna.create_study(direction='maximize')

        # Optimize the objective function
        study.optimize(
            partial(
                tuning.objective_function,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                pipeline=pipeline,
                param_space=RFC_SPACE
            ),
            n_trials=10,
            callbacks=[tuning.champion_callback]
        )

        best_params = study.best_params
        print("BEST PARAMS FROM main(): ", best_params)

        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = tuning.get_classification_metrics(y_test, y_pred, prefix='best_model_test')

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "best_model")
""".lstrip()

    # Define the filename for the new script
    filename = "generated_script.py"
    
    # Write the script content to the new file
    with open(filename, "w") as file:
        file.write(script_content)
    
    print(f"Script '{filename}' has been generated.")