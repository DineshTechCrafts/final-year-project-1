import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.normalization import FeatureNormalizer

def test_scaler_leakage():
    normalizer = FeatureNormalizer(["feat1", "feat2"])
    
    train_df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat2": [10.0, np.nan, 30.0, 40.0, 50.0]
    })
    
    test_df_1 = pd.DataFrame({
        "feat1": [100.0, 200.0],
        "feat2": [np.nan, 1000.0]
    })
    
    test_df_2 = pd.DataFrame({
        "feat1": [-100.0, -200.0],
        "feat2": [np.nan, -1000.0]
    })
    
    # Fit only on train
    normalizer.fit(train_df)
    
    # Transform test sets separately
    out1 = normalizer.transform(test_df_1)
    out2 = normalizer.transform(test_df_2)
    
    # Ensure imputation used the TRAIN median (which is 35 for feat2)
    # So the scaled value for NaN should be the scaled value of 35
    # If the scaler leaked, test_df_1's NaN would be imputed using 1000, 
    # and test_df_2's NaN would be imputed using -1000.
    
    # feat2 missing indicator should be 1
    assert out1["feat2_missing"].iloc[0] == 1
    assert out2["feat2_missing"].iloc[0] == 1
    
    # The imputed and scaled value for feat2 should be IDENTICAL in out1 and out2,
    # because it comes purely from the train fit.
    assert np.isclose(out1["feat2"].iloc[0], out2["feat2"].iloc[0])
