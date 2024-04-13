''' Docstring
'''
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from starter.ml import data, model

# Add code to load in the data.
train = pd.read_csv(
    '/home/chafund/GIT/Deploying-a-ML-Model-to-Cloud-Application'
    '-Platform-with-FastAPI/data/train.csv'
    )

test = pd.read_csv(
    '/home/chafund/GIT/Deploying-a-ML-Model-to-Cloud-Application'
    '-Platform-with-FastAPI/data/test.csv'
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
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
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
