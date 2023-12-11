# Standard imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, recall_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split

# Astro imports

"""
Train a Random Forest Classifier on star/galaxy truth coordinates.
This model is then used in photometry.py to generate the ePSF. 
"""

def train_model():
    """
    Inputs: flux, max pixel, ellipticity for galaxies and stars. 
    1. Source detection on images at truth coordinates. 
    Wait, no, you need ellipticty at truth coordinates. Does SEP do forced source detection? 
    """