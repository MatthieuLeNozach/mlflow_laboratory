# file: hyperparameter_bank.py

################### CLASSIFICATION ###################

# Hyperparameters for KNearestNeighborsClassifier
KNNC_SPACE = {
    'classifier__n_neighbors': (1, 30),
    'classifier__weights': ['uniform', 'distance'],
    'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'classifier__leaf_size': (10, 50),
    'classifier__p': [1, 2],
    'classifier__metric': ['euclidean', 'manhattan', 'minkowski'],
}


# Hyperparameters for RandomForestClassifier
RFC_SPACE = {
    'classifier__n_estimators': (20, 200),
    'classifier__max_depth': (10, 100),
    'classifier__min_samples_split': (2, 20),
    'classifier__min_samples_leaf': (1, 2),
    'classifier__max_features': ['sqrt', 'log2', None],
    'classifier__criterion': ['gini', 'entropy'],
}


# Hyperparameters for SGDClassifier
SGDC_SPACE = {
    'classifier__alpha': (1e-4, 1e-2, 'log'),
    'classifier__penalty': ['l2', 'l1', 'elasticnet'],
    # ...
}

# Hyperparameters for SupportVectorClassifier
SVC_SPACE = {
    'classifier__C': (0.1, 10, 'log'),
    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'classifier__degree': (2, 5),
    'classifier__gamma': (1e-4, 1, 'log'),
    'classifier__coef0': (0, 1, 'log'),
    'classifier__shrinking': [True, False],
    'classifier__tol': (1e-5, 1e-2, 'log'),
}



################### REGRESSION ###################


RFR_SPACE = {
    'regressor__n_estimators': (20, 200),
    'regressor__max_depth': (10, 100),
    'regressor__min_samples_split': (2, 20),
    'regressor__min_samples_leaf': (1, 2),
    'regressor__max_features': ['sqrt', 'log2', None],
    'regressor__criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
}
# ...