# file: pipeline_utils.py

from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import inspect

num_features = []
cat_features = []
classifier = None


BASE_PREPROCESSING_PIPELINE = ColumnTransformer(
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


BINARY_CLASSIFICATION_PIPELINE = Pipeline(steps=[
    ('preprocessing', BASE_PREPROCESSING_PIPELINE),
    ('classifier', classifier)
])

