# file: pipeline_utils.py

from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Optional, Iterable

def get_preprocessing_pipeline(
    num_features: List[str], cat_features: Optional[List[str]] = []
) -> Pipeline:
    """Makes a simple numerical / categorical pipeline"""
    
    preprocessing = ColumnTransformer(
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
    return Pipeline(steps=[('preprocessing', preprocessing)])


def get_sklearn_classifier_pipeline(
    num_features: List[str], 
    cat_features: Optional[List[str]] = [], 
    classifier: ClassifierMixin = None,
) -> Pipeline:
    """Creates a preprocessing / classification pipeline with a scikit-learn classifier"""
    
    if classifier is None:
        raise ValueError('A classifier instance must be provided')
    
    preprocessing_pipeline = get_preprocessing_pipeline(num_features, cat_features)
    
    classification_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('classifier', classifier)
    ])
    return classification_pipeline



def get_binary_rfc_pipeline(
    num_features: List[str], cat_features: Optional[List[str]] = [],
) -> Pipeline:
    """Creates a classification pipeline with preprocessing and RFC"""
    
    preprocessing_pipeline = get_preprocessing_pipeline(num_features, cat_features)
    
    classification_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('classifier', RandomForestClassifier())
    ])
    return classification_pipeline