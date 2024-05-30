# Shopping

## KNN Class

This is a custom KNN classifier that uses bagging to improve performance.

Attributes:

k: Number of neighbors to use.

n_estimators: Number of bagging estimators.

max_samples: Maximum proportion of samples to draw for each base estimator.

Methods:

fit(X, y): Fits the model using the training data X and labels y. It resamples the data for each estimator and fits a simple KNN model to each sample.

predict(X): Predicts the labels for the input data X by aggregating predictions from all the base estimators and using majority voting.

## MySimpleKNN Class

This is a simplified version of a KNN model based on KDTree for efficient neighbor searches.

Attributes:
k: Number of neighbors to use.

tree: KDTree for efficient neighbor searches.

Methods:

fit(X, y): Fits the KDTree to the training data X and labels y.

predict(X): Predicts the labels for the input data X by finding the k nearest neighbors and using majority voting to determine the label.

## Other Functions
load_data(filename): Loads and preprocesses data from a CSV file.

train_model(evidence, labels): Trains the KNN model using the provided evidence and labels.

evaluate(labels, predictions): Evaluates the model's predictions by calculating sensitivity, specificity, and F1 score.
