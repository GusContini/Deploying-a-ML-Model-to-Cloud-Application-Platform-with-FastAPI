# Deploying-a-ML-Model-to-Cloud-Application-Platform-with-FastAPI

Development of a classification model on publicly available Census Bureau data. Here we will create unit tests to monitor the model performance on various data slices.

Then, we will deploy our model using the FastAPI package and create API tests.

The slice validation and the API tests will be incorporated into a CI/CD framework using GitHub Actions.

__________________________________________________________________________________________________________________________________

Run the following command to start the API:
> uvicorn main:app

Then go to this URL to access the API documentation: http://127.0.0.1:8000/docs

This is how it looks like:

image.png

There one can try out the endpoints. Here's the output of a post/predict test:

image.png