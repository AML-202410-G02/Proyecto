import Definitions

import os.path as osp
import pandas as pd

from joblib import load
from src.model.DataPreprocessor import DataPreprocessor


class ModelController:

    def __init__(self):
        self.model_path = osp.join('resources','models','model.joblib')
        if not osp.exists(self.model_path):
            raise FileNotFoundError(f"El archivo no fue encontrado en {self.model_path}")
        self.model = load(self.model_path)

        self.preprocessor = DataPreprocessor()

    def predict(self, data):

        x = self.model.named_steps['preprocessor'].transform(data)
        y_pred = self.model.named_steps['classifier'].predict(x)
        
        return y_pred
