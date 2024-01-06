import numpy as np

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        # Initialize weights for each sample
        n_samples = len(X)
        sample_weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # Create a weak learner (Decision Stump in this case)
            weak_learner = DecisionStump()
            
            # Train weak learner with weighted samples
            weak_learner.fit(X, y, sample_weights)
            
            # Make predictions
            predictions = weak_learner.predict(X)
            
            # Compute weighted error
            weighted_error = np.sum(sample_weights * (predictions != y))
            
            # Calculate the alpha value (weight of the weak learner in the final model)
            alpha = 0.5 * np.log((1 - weighted_error) / weighted_error)
            self.alphas.append(alpha)
            
            # Update sample weights
            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)  # Normalize
            
            # Save the weak learner
            self.models.append(weak_learner)

    def predict(self, X):
        # Combine the predictions of all weak learners
        predictions = np.sum(alpha * model.predict(X) for alpha, model in zip(self.alphas, self.models))
        return np.sign(predictions)


class DecisionStump:
    def __init__(self, vocab):
        self.feature_index = None
        self.threshold = None
        self.prediction = None
        

    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        min_error = float('inf')

        for feature_index in range(n_features):
            unique_thresholds = np.unique(X[:, feature_index])
            for threshold in unique_thresholds:
                predictions = np.ones(n_samples)
                predictions[X[:, feature_index] < threshold] = -1

                error = np.sum(sample_weights * (predictions != y))

                if error < min_error:
                    min_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.prediction = 1 if np.sum(sample_weights * y * predictions) > 0 else -1

    def predict(self, X):
        predictions = np.ones(len(X))
        predictions[X[:, self.feature_index] < self.threshold] = -1
        return predictions


# Example usage:
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a synthetic dataset for demonstration
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create AdaBoost classifier
adaboost_classifier = AdaBoost(n_estimators=50)

# Train the AdaBoost classifier
adaboost_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = adaboost_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
