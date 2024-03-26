# file: artifacts.py

import os
import io
import tempfile
import pandas as pd
import mlflow

from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def log_artifact_with_buffer(buffer, artifact_name):
    """Log a file-like object (buffer) as an artifact in MLflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = os.path.join(tmpdir, artifact_name)
        with open(temp_path, 'wb') as f:
            f.write(buffer.getbuffer())
        mlflow.log_artifact(temp_path)

def log_roc_curve(y_true, y_pred_proba, set_name=''):
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]  # Take the second column for binary classification
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_pred_proba, ax=ax)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    
    log_artifact_with_buffer(buf, f'roc_curve_{set_name}.png')
    buf.close()
    
def log_confusion_matrix(y_true, y_pred, set_name=''):
    cm = pd.crosstab(y_true, y_pred, rownames=['y_true'], colnames=['y_pred'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap='RdBu_r', fmt='g')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    
    log_artifact_with_buffer(buf, f'confusion_matrix_{set_name}.png')
    buf.close()