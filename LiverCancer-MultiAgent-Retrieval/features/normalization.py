import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle
import os

class FeatureNormalizer:
    def __init__(self, numerical_features):
        self.numerical_features = numerical_features
        # Median imputation
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame):
        """
        Fits imputer and scaler purely on the non-missing values of the DataFrame.
        Expected to be called ONLY on the TRAIN partition.
        """
        data = df[self.numerical_features].copy()
        
        # Fit imputer
        self.imputer.fit(data)
        data_imputed = self.imputer.transform(data)
        
        # Fit scaler
        self.scaler.fit(data_imputed)
        
        self.is_fitted = True
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the dataframe, adding _missing indicator columns.
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted on TRAIN before calling transform.")
            
        out_df = df.copy()
        
        for col in self.numerical_features:
            if col in out_df.columns:
                # Add missing indicator BEFORE imputation
                out_df[f"{col}_missing"] = out_df[col].isna().astype(int)
        
        # Impute
        data = out_df[self.numerical_features]
        data_imputed = self.imputer.transform(data)
        
        # Scale
        data_scaled = self.scaler.transform(data_imputed)
        
        out_df[self.numerical_features] = data_scaled
        
        return out_df
        
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "imputer": self.imputer,
                "scaler": self.scaler,
                "numerical_features": self.numerical_features
            }, f)
            
    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.imputer = state["imputer"]
        self.scaler = state["scaler"]
        self.numerical_features = state["numerical_features"]
        self.is_fitted = True
