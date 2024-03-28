# file: binary_classification.py

# FIND REGULAR IMPORTS IN laboratory/config.py
from laboratory.config import *
from laboratory.mlflow import get_or_create_experiment, set_mlflow_tracking_uri_from_env

handle_warnings()
env_vars = get_environment()
set_mlflow_tracking_uri_from_env(env_vars)

# Custom modules
import laboratory.dataset as dataset
import laboratory.sklearn as sklearn
import laboratory.tuning as tuning
from laboratory.mlflow import get_run_name
from laboratory.artifacts import log_confusion_matrix, log_roc_curve
from lib.models.sklearn import RandomForestClassifier # PLACEHOLDER, only needed when used as executable script

###############################################
#################### SETUP ####################
######## Fill experiment input here ########### 

DATASET_PATH = DATASET_PATH # PLACEHOLDER, fill only when used as executable script
DF = pd.read_csv(DATASET_PATH)

TARGET_NAME = TARGET_NAME # Another PLACEHOLDER...
FEATURES_TO_DROP = FEATURES_TO_DROP # PLACEHOLDER

DATASET_SPLIT_PARAMS = {'test_size': 0.2, 'stratify': DF[TARGET_NAME], 'random_state': 42} # or stratify=None

CLASSIFIER = clf # PLACEHOLDER
SPACE = RFC_SPACE # PLACEHOLDER
SAVE_MODEL = False

OPTUNA_STUDY_TRIALS = 20
OPTUNA_METRIC_TO_MAXIMIZE = 'f1_score'  


EXPERIMENT_NAME = DATASET_PATH.split('/')[-1] # Experiment name is file name by default
RUN_NAME = get_run_name(run_name=None) # Run name is date+time by default


###############################################
#################### MAIN #####################

if __name__ == '__main__':
    df = DF.copy()
    
    if FEATURES_TO_DROP is not None:
        df = df.drop(columns=FEATURES_TO_DROP)    
        
    
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=TARGET_NAME), df[TARGET_NAME], **DATASET_SPLIT_PARAMS
    )
    
    num_features = X_train.select_dtypes([np.number]).columns.tolist()
    cat_features = X_train.columns.difference(num_features).tolist()

    PREPROCESSING_PIPELINE = ColumnTransformer(
        transformers=[
            ('numerical', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_features),
            ('categorical', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder())
            ]), cat_features)
        ]
    )

    CLASSIFICATION_PIPELINE = Pipeline(steps=[
        ('preprocessing', PREPROCESSING_PIPELINE),
        ('classifier', CLASSIFIER)
    ])

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
                pipeline=CLASSIFICATION_PIPELINE,
                param_space=SPACE,
                metric_to_maximize=OPTUNA_METRIC_TO_MAXIMIZE
            ),
            n_trials=OPTUNA_STUDY_TRIALS,
            callbacks=[tuning.champion_callback]
        )

        best_params = study.best_params
        print("BEST PARAMS FROM main(): ", best_params)

        CLASSIFICATION_PIPELINE.set_params(**best_params)
        CLASSIFICATION_PIPELINE.fit(X_train, y_train)
        y_pred = CLASSIFICATION_PIPELINE.predict(X_test)
        y_pred_proba = CLASSIFICATION_PIPELINE.predict_proba(X_test)

        metrics = tuning.get_classification_metrics(y_test, y_pred, y_pred_proba)
        log_confusion_matrix(y_test, y_pred)
        log_roc_curve(y_test, y_pred_proba)

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        if SAVE_MODEL:
            mlflow.sklearn.log_model(sklearn, "best_model")