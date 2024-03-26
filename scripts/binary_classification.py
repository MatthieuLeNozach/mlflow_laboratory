# file: binary_classification.py

# FIND REGULAR IMPORTS IN laboratory/config.py
from laboratory.config import *
from laboratory.mlflow import get_or_create_experiment, set_mlflow_tracking_uri_from_env

handle_warnings()
env_vars = get_environment()
set_mlflow_tracking_uri_from_env(env_vars)

# Custom modules
import laboratory.dataset as dataset
import laboratory.pipeline as pipeline
import laboratory.tuning as tuning
from laboratory.artifacts import log_confusion_matrix, log_roc_curve
from lib.hp.sklearn import RFC_SPACE
from lib.models.sklearn import RandomForestClassifier


#################### SETUP ####################
EXPERIMENT_NAME = 'optuna_experiment14'
RUN_NAME = 'hyperparameter_optimization'
DATASET_PATH = '../../0_DATASETS/creditcard.csv'
CLASSIFIER = RandomForestClassifier(bootstrap=False)
SPACE = RFC_SPACE 
TARGET_NAME = 'Class'
SAVE_MODEL = False

#################### MAIN ####################

if __name__ == '__main__':
    df = pd.read_csv(DATASET_PATH)
    df_redux = dataset.data_split_redux(df, TARGET_NAME, zero_label_redux=0.995)
    
    X_train, X_test, y_train, y_test = dataset.train_test_split(
        df_redux.drop(columns=TARGET_NAME), df_redux[TARGET_NAME], test_size=0.2, random_state=42
    )
    
    num_features = X_train.select_dtypes([np.number]).columns.tolist()
    cat_features = X_train.columns.difference(num_features).tolist()
    
    pipeline = pipeline.get_binary_rfc_pipeline(num_features, cat_features)
        
    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME) # DEBUGGING
    print(f"Experiment: {experiment.name}, ID: {experiment.experiment_id}") # DEBUGGING
    
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
        y_pred_proba = pipeline.predict_proba(X_test)

        metrics = tuning.get_classification_metrics(y_test, y_pred, y_pred_proba, prefix='best_model_test')
        log_confusion_matrix(y_test, y_pred)
        log_roc_curve(y_test, y_pred_proba)

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        if SAVE_MODEL:
            mlflow.sklearn.log_model(pipeline, "best_model")



 