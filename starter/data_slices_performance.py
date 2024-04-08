import pandas as pd
import sys
sys.path.append('ml')  # Adiciona o subdiretório X2 ao caminho de busca de módulos
import model
import data
import json

def compute_performance_slices(feature):

    test = pd.read_csv(
        '/home/chafund/GIT/Deploying-a-ML-Model-to-Cloud-Application-Platform-with-FastAPI/data/test.csv'
        )

    # Get unique values of the specified feature
    unique_values = test[feature].unique()

    # Initialize dictionary to store performance metrics for each slice
    metrics_dict = {}

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

    model_path = '/home/chafund/GIT/Deploying-a-ML-Model-to-Cloud-Application-Platform-with-FastAPI/starter/trained_model.pkl'

    trained_model, trained_encoder, trained_lb = model.import_model(model_path)

    # Iterate over unique feature values
    for value in unique_values:
        # Select data slice with the specified feature value
        slice_indices = test[feature] == value
        data_slice = test[slice_indices]

        X_test, y_test, _, _ = data.process_data(
            data_slice, categorical_features=cat_features, label="salary",
            training=False, encoder=trained_encoder, lb=trained_lb
            )

        # Make predictions using the model on the slice
        predictions = model.inference(trained_model, X_test)

        # Compute performance metrics for the slice
        precision, recall, fbeta = model.compute_model_metrics(y_test, predictions)

        # Store performance metrics in the dictionary
        metrics_dict[value] = {'precision': precision, 'recall': recall, 'fbeta': fbeta}

        with open('data_slice_model_metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)

    return metrics_dict

if __name__ == '__main__':
    compute_performance_slices('relationship')