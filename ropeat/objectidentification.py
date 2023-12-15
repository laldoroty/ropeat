# Standard imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, recall_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Internal imports
from .utils import train_config

"""
Train a Random Forest Classifier on star/galaxy truth coordinates.
This model is then used in photometry.py to generate the ePSF. 
"""

def train_model(train_table,test_size=0.4,random_state=42):
    """
    Inputs: flux, max pixel, ellipticity for galaxies and stars. 
    Astropy table with sum of flux in pixels, max pixel value, ellipticity, and object type.

    
    """

    train_table = train_table.to_pandas()
    X = train_table[['peak','flux','ellipticity']]
    y = np.array(list(map(train_config,train_table['objtype'])))

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)

    model = RandomForestClassifier(max_depth=100, n_estimators=100, max_features=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    data = {'X': X, 'y': y, 
            'X_train': X_train, 'X_test': X_test, 
            'y_train': y_train, 'y_test': y_test,
            'y_pred': y_pred}
    scores = {'score': score, 'accuracy': accuracy, 'precision': precision, 'recall': recall}

    return model, data, scores