# Script to train machine learning model.

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sys
sys.path.append('ml')  # Adiciona o subdiretório X2 ao caminho de busca de módulos
import data
import model
import pickle

# Add code to load in the data.
#     '\\\\wsl.localhost\\Ubuntu\\home\\chafund\\GIT\\Deploying-a-ML-Model-to-Cloud-Application-Platform-with-FastAPI\\data\\census.csv'
train = pd.read_csv(
    '/home/chafund/GIT/Deploying-a-ML-Model-to-Cloud-Application-Platform-with-FastAPI/data/train.csv'
    )

test = pd.read_csv(
    '/home/chafund/GIT/Deploying-a-ML-Model-to-Cloud-Application-Platform-with-FastAPI/data/test.csv'
    )

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = data.process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = data.process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=3,
    n_jobs=-1,
    criterion='gini',
    max_features=0.5,
    oob_score=True,
    random_state=42
)

clf = model.train_model(X_train, y_train, rf)

pred = model.inference(clf, X_test)

precision, recall, fbeta = model.compute_model_metrics(y_test, pred)
print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'fbeta: {fbeta}')

with open('trained_model.pkl', 'wb') as file:
    pickle.dump((clf, encoder, lb), file)