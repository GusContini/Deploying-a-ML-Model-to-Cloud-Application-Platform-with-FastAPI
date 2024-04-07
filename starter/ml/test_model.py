import pytest
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os

# Import functions to be tested
from model import train_model, compute_model_metrics, inference, import_model

# Test if train_model function returns trained model of the expected type
def test_train_model():
    # Create dummy data
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    rf = RandomForestClassifier()
    
    # Call train_model function
    trained_model = train_model(X_train, y_train, rf)
    
    # Assert that trained_model is an instance of RandomForestClassifier
    assert isinstance(trained_model, RandomForestClassifier)

def test_compute_model():
    # Create dummy data
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    
    # Call compute_model_metrics function
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    # Assert that performance metrics are of expected types
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

# Test if inference function returns predictions of the expected type
def test_inference():
    # Create dummy data
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    model = RandomForestClassifier()
    model.fit(X_train, y_train)  # Train the model
    
    # Create dummy data for inference
    X = np.array([[1, 2], [3, 4]])
    
    # Call inference function
    predictions = inference(model, X)
    
    # Assert that predictions is a numpy array
    assert isinstance(predictions, np.ndarray)

# Run tests
if __name__ == '__main__':
    test_train_model()
    test_compute_model()
    test_inference()
