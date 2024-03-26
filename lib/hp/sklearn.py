# file: hyperparameter_bank.py

################### CLASSIFICATION ###################

# Hyperparameters for DecisionTreeClassifier
DTC_SPACE = {
 'classifier__criterion': ['gini', 'entropy'],
 'classifier__splitter': ['best', 'random'],
 'classifier__max_depth': (2, 20),
 'classifier__min_samples_split': (2, 20),
 'classifier__min_samples_leaf': (1, 20),
 'classifier__max_features': [None, 'sqrt', 'log2'],
 'classifier__class_weight': [None, 'balanced'],
 'classifier__ccp_alpha': (0, 1)
}

# Hyperparameters for GradientBoostingClassifier
GBC_SPACE = {
 'classifier__loss': ['deviance', 'exponential'],
 'classifier__learning_rate': (1e-3, 1, 'log'),
 'classifier__n_estimators': (50, 500),
 'classifier__subsample': (0.5, 1),
 'classifier__criterion': ['friedman_mse', 'squared_error'],
 'classifier__min_samples_split': (2, 20),
 'classifier__min_samples_leaf': (1, 20),
 'classifier__max_depth': (2, 10),
 'classifier__max_features': [None, 'sqrt', 'log2'],
 'classifier__max_leaf_nodes': (2, 50),
 'classifier__min_impurity_decrease': (0, 1),
 'classifier__ccp_alpha': (0, 1)
}

# Hyperparameters for KNeighborsClassifier
KNC_SPACE = {
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


# Hyperparameters for SGDClassifier NO PROBABILITIES
SGDC_SPACE = {
 'classifier__loss': ['hinge', 'squared_hinge', 'perceptron'],
 'classifier__penalty': ['l2', 'l1', 'elasticnet'],
 'classifier__alpha': (1e-7, 1e-1, 'log'),
 'classifier__l1_ratio': (0, 1),
 'classifier__fit_intercept': [True, False],
 'classifier__max_iter': (100, 1000),
 'classifier__tol': (1e-5, 1e-2, 'log'),
 'classifier__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
 'classifier__eta0': (1e-7, 1e-1, 'log'),
 'classifier__power_t': (0.1, 0.5)
}

SGDC_SPACE_PROBA = {
 'classifier__loss': ['log_loss', 'modified_huber'],
 'classifier__penalty': ['l2', 'l1', 'elasticnet'],
 'classifier__alpha': (1e-7, 1e-1, 'log'),
 'classifier__l1_ratio': (0, 1),
 'classifier__fit_intercept': [True, False],
 'classifier__max_iter': (100, 1000),
 'classifier__tol': (1e-5, 1e-2, 'log'),
 'classifier__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
 'classifier__eta0': (1e-7, 1e-1, 'log'),
 'classifier__power_t': (0.1, 0.5)
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