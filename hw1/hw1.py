from ucimlrepo import fetch_ucirepo 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Function to split dataset into training and testing sets
def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the dataset into training and testing sets
    
    Parameters:
    X: features
    y: target labels
    test_size: proportion of the dataset to include in the test split
    random_state: random seed for reproducibility
    
    Returns:
    X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_samples = int(n_samples * test_size)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Function to calculate confusion matrix
def confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix
    
    Parameters:
    y_true: true labels
    y_pred: predicted labels
    
    Returns:
    confusion matrix
    """
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i in range(len(y_true)):
        true_idx = np.where(classes == y_true[i])[0][0]
        pred_idx = np.where(classes == y_pred[i])[0][0]
        cm[true_idx, pred_idx] += 1
        
    return cm

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes):
    """
    Plot confusion matrix
    
    Parameters:
    cm: confusion matrix
    classes: class labels
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_test, y_prob):
    """
    Plot ROC curve
    
    Parameters:
    y_test: true labels
    y_prob: predicted probabilities
    """
    plt.figure(figsize=(8, 6))
    
    # Convert to binary classification
    y_test_binary = (y_test == np.max(y_test)).astype(int)
    y_prob_binary = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob
    
    fpr, tpr, _ = roc_curve(y_test_binary, y_prob_binary)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Function to plot ROC curves comparison
def plot_roc_curves_comparison(y_test, nb_probs, knn_probs):
    """
    Compare ROC curves of two models
    
    Parameters:
    y_test: true labels
    nb_probs: Naive Bayes predicted probabilities
    knn_probs: KNN predicted probabilities
    """
    plt.figure(figsize=(10, 8))
    
    # Convert to binary classification
    y_test_binary = (y_test == np.max(y_test)).astype(int)
    
    # Calculate ROC for Naive Bayes
    nb_prob = nb_probs[:, 1] if nb_probs.shape[1] > 1 else nb_probs
    nb_fpr, nb_tpr, _ = roc_curve(y_test_binary, nb_prob)
    nb_roc_auc = auc(nb_fpr, nb_tpr)
    
    # Calculate ROC for KNN
    knn_prob = knn_probs[:, 1] if knn_probs.shape[1] > 1 else knn_probs
    knn_fpr, knn_tpr, _ = roc_curve(y_test_binary, knn_prob)
    knn_roc_auc = auc(knn_fpr, knn_tpr)
    
    # Plot ROC curves
    plt.plot(nb_fpr, nb_tpr, color='darkorange', lw=2, 
             label=f'Naive Bayes ROC curve (AUC = {nb_roc_auc:.2f})')
    plt.plot(knn_fpr, knn_tpr, color='green', lw=2, 
             label=f'KNN (k=1) ROC curve (AUC = {knn_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.show()

class NaiveBayes:
    """
    Naive Bayes classifier implementation
    """
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
        
    def fit(self, X, y):
        """
        Train the Gaussian Naive Bayes classifier
        
        Parameters:
        X: feature matrix
        y: label vector
        
        Returns:
        self: fitted model
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize matrices for mean, variance and prior probabilities
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        # Calculate mean, variance and prior probabilities for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-10  # Avoid zero variance
            self.priors[idx] = X_c.shape[0] / n_samples
            
        return self
    
    def _calculate_likelihood(self, x, mean, var):
        """
        Calculate Gaussian probability density function
        
        Parameters:
        x: feature vector
        mean: mean vector
        var: variance vector
        
        Returns:
        probability density
        """
        # Gaussian PDF: 1/sqrt(2*pi*var) * exp(-(x-mean)^2 / (2*var))
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return exponent / np.sqrt(2 * np.pi * var)
    
    def _calculate_posterior(self, x):
        """
        Calculate posterior probabilities
        
        Parameters:
        x: feature vector
        
        Returns:
        posterior probabilities
        """
        posteriors = []
        
        # Calculate posterior probability for each class
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            likelihood = np.sum(np.log(self._calculate_likelihood(x, self.mean[idx, :], self.var[idx, :])))
            posterior = prior + likelihood
            posteriors.append(posterior)
            
        return posteriors
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        X: feature matrix
        
        Returns:
        predicted labels
        """
        y_pred = []
        for x in X:
            posteriors = self._calculate_posterior(x)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Parameters:
        X: feature matrix
        y: true labels
        
        Returns:
        accuracy score
        """
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy
    
    def predict_proba(self, X):
        """
        Calculate prediction probabilities for each class
        
        Parameters:
        X: feature matrix
        
        Returns:
        probability matrix
        """
        probas = []
        for x in X:
            posteriors = self._calculate_posterior(x)
            # Convert log probabilities to normal probabilities and normalize
            probs = np.exp(posteriors - np.max(posteriors))
            probs /= np.sum(probs)
            probas.append(probs)
        return np.array(probas)
    
class KNNClassifier:
    """
    k-Nearest Neighbors classifier implementation
    """
    def __init__(self, k=1):
        """
        Initialize KNN classifier
        
        Parameters:
        k: number of neighbors, default is 1
        """
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Store training data
        
        Parameters:
        X: feature matrix
        y: label vector
        
        Returns:
        self: fitted model
        """
        self.X_train = X
        self.y_train = y
        return self
    
    def _euclidean_distance(self, x1, x2):
        """
        Calculate Euclidean distance between two vectors
        
        Parameters:
        x1, x2: feature vectors
        
        Returns:
        distance
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict_single(self, x):
        """
        Predict class for a single sample
        
        Parameters:
        x: feature vector
        
        Returns:
        predicted class
        """
        # Calculate distances between test sample and all training samples
        distances = np.array([self._euclidean_distance(x, x_train) for x_train in self.X_train])
        
        # Find indices of k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        
        # Get the classes of these samples
        k_nearest_labels = self.y_train[k_indices]
        
        # For k=1, simply return the class of the nearest sample
        return k_nearest_labels[0]
    
    def predict(self, X):
        """
        Predict classes for multiple samples
        
        Parameters:
        X: feature matrix
        
        Returns:
        predicted classes
        """
        return np.array([self.predict_single(x) for x in X])
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Parameters:
        X: feature matrix
        y: true labels
        
        Returns:
        accuracy score
        """
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy
    
    def predict_proba(self, X):
        """
        Predict probabilities (for k=1, probabilities are either 0 or 1)
        
        Parameters:
        X: feature matrix
        
        Returns:
        probability matrix
        """
        y_pred = self.predict(X)
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        n_samples = X.shape[0]
        
        # Initialize probability matrix
        probas = np.zeros((n_samples, n_classes))
        
        # For each prediction, set probability to 1 for the predicted class
        for i, pred in enumerate(y_pred):
            class_idx = np.where(classes == pred)[0][0]
            probas[i, class_idx] = 1.0
            
        return probas

# Main execution code
if __name__ == "__main__":
    # # Fetch Breast Cancer dataset 
    # breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
      
    # # Get data (as pandas dataframes) 
    # X = breast_cancer_wisconsin_diagnostic.data.features 
    # y = breast_cancer_wisconsin_diagnostic.data.targets

    # # fetch dataset 
    # ionosphere = fetch_ucirepo(id=52) 
    
    # # Get data (as pandas dataframes) 
    # X = ionosphere.data.features 
    # y = ionosphere.data.targets 

    # # fetch dataset 
    # iris = fetch_ucirepo(id=53) 
    
    # # data (as pandas dataframes) 
    # X = iris.data.features 
    # y = iris.data.targets 

    # fetch dataset 
    wine = fetch_ucirepo(id=109) 
    
    # data (as pandas dataframes) 
    X = wine.data.features 
    y = wine.data.targets 
    
    # Convert data to numpy arrays
    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.6, random_state=42
    )
    
    # Build and train Naive Bayes model
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    
    # Evaluate Naive Bayes model
    accuracy = nb.score(X_test, y_test)
    print(f"Naive Bayes accuracy: {accuracy:.4f}")
    
    # Make predictions with Naive Bayes
    y_pred = nb.predict(X_test)
    
    # Calculate confusion matrix for Naive Bayes
    cm = confusion_matrix(y_test, y_pred)
    print("Naive Bayes confusion matrix:")
    print(cm)
    
    # Plot confusion matrix for Naive Bayes
    unique_classes = np.unique(y_test)
    plot_confusion_matrix(cm, unique_classes)
    
    # Train and evaluate KNN classifier
    print("\n===== 1-Nearest Neighbor Classifier =====")
    knn = KNNClassifier(k=1)
    knn.fit(X_train, y_train)
    
    # Evaluate KNN model
    knn_accuracy = knn.score(X_test, y_test)
    print(f"KNN accuracy: {knn_accuracy:.4f}")
    
    # Make predictions with KNN
    knn_y_pred = knn.predict(X_test)
    
    # Calculate confusion matrix for KNN
    knn_cm = confusion_matrix(y_test, knn_y_pred)
    print("KNN confusion matrix:")
    print(knn_cm)
    
    # Plot confusion matrix for KNN
    plot_confusion_matrix(knn_cm, unique_classes)
    
    # exit()
    # Calculate predicted probabilities for Naive Bayes
    nb_y_prob = nb.predict_proba(X_test)

    # Calculate predicted probabilities for KNN
    knn_y_prob = knn.predict_proba(X_test)
    
    # Compare ROC curves
    plot_roc_curves_comparison(y_test, nb_y_prob, knn_y_prob)
    
    # Compare model accuracies
    print("\n===== Model Comparison =====")
    print(f"Naive Bayes accuracy: {accuracy:.4f}")
    print(f"KNN (k=1) accuracy: {knn_accuracy:.4f}")


