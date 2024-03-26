# file: config.py

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import optuna
from functools import partial
import mlflow

def get_root_abs_path():
    return Path(__file__).resolve().parent.parent

    
def get_environment():
    import os
    print(os.getcwd())
    print(os.listdir())
    from pathlib import Path
    from dotenv import load_dotenv

    # Load environment variables from .environment/.env and .environment/.env.secret
    env_path = get_root_abs_path() / '.environment'
    load_dotenv(env_path / '.env')
    load_dotenv(env_path / '.env.secret')

    print("MINIO_ENDPOINT_URL:", 'set' if os.getenv('MINIO_ENDPOINT_URL') else 'not set')
    print("MINIO_ACCESS_KEY:", 'set' if os.getenv('MINIO_ACCESS_KEY') else 'not set')
    print("MINIO_SECRET_KEY:", 'set' if os.getenv('MINIO_SECRET_KEY') else 'not set')
    
    return os.environ

def handle_warnings():
    import warnings
    warnings.filterwarnings("ignore")
