''' Docstring
'''
import pickle
import json
from sklearn.metrics import fbeta_score, precision_score, recall_score

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, model):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    clf = model.fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)

    metrics = {
        "precision": precision,
        "recall": recall,
        "fbeta": fbeta
    }

    with open('model_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : pickle file
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def import_model(model_path):
    """ Import trained model, onehot encoder and label binarizer.

    Inputs
    ------
    model : pickle file
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Pre
    """
    with open(model_path, 'rb') as f:
        trained_model, encoder, lb = pickle.load(f)
    return trained_model, encoder, lb
