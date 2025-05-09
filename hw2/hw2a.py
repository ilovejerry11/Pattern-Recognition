from ucimlrepo import fetch_ucirepo 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc


if __name__ == "__main__":
    # # Fetch Breast Cancer dataset 
    # breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
      
    # # Get data (as pandas dataframes) 
    # X = breast_cancer_wisconsin_diagnostic.data.features.values 
    # y = breast_cancer_wisconsin_diagnostic.data.targets.values.ravel()

    # fetch dataset 
    ionosphere = fetch_ucirepo(id=52) 
    
    # Get data (as pandas dataframes) 
    X = ionosphere.data.features.values 
    y = ionosphere.data.targets.values.ravel() 

    # # fetch dataset 
    # iris = fetch_ucirepo(id=53) 
    
    # # data (as pandas dataframes) 
    # X = iris.data.features.values
    # y = iris.data.targets.values.ravel()

    # # fetch dataset 
    # wine = fetch_ucirepo(id=109) 
    
    # # data (as pandas dataframes) 
    # X = wine.data.features.values
    # y = wine.data.targets.values.ravel()

    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split into training and testing sets (8:2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compute Sw and Sb using training data
    mean_overall = np.mean(X_train, axis=0)
    classes = np.unique(y_train)
    Sw = np.zeros((X_train.shape[1], X_train.shape[1]))
    Sb = np.zeros((X_train.shape[1], X_train.shape[1]))
    for c in classes:
        Xc = X_train[y_train == c]
        mean_c = np.mean(Xc, axis=0)
        Sw += np.dot((Xc - mean_c).T, (Xc - mean_c))
        n_c = Xc.shape[0]
        mean_diff = (mean_c - mean_overall).reshape(-1, 1)
        Sb += n_c * np.dot(mean_diff, mean_diff.T)

    # Separability before projection (training data)
    separability_before = np.trace(Sb) / np.trace(Sw)
    print(f"Separability before projection (train): {separability_before:.4f}")

    # LDA: Find projection vector w using training data
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))
    w = eigvecs[:, np.argmax(eigvals)].real

    # Project both train and test data
    X_train_lda = X_train @ w
    X_test_lda = X_test @ w

    # Separability after projection (training data)
    mean_overall_lda = np.mean(X_train_lda)
    Sw_lda = 0
    Sb_lda = 0
    for c in classes:
        Xc_lda = X_train_lda[y_train == c]
        mean_c_lda = np.mean(Xc_lda)
        Sw_lda += np.sum((Xc_lda - mean_c_lda) ** 2)
        n_c = Xc_lda.shape[0]
        Sb_lda += n_c * (mean_c_lda - mean_overall_lda) ** 2
    separability_after = Sb_lda / Sw_lda
    print(f"Separability after projection (train): {separability_after:.4f}")

    # Naive Bayes classification on test data (before projection)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Naive Bayes accuracy on test data (before projection): {acc:.4f}")

    # Naive Bayes classification on test data (after projection)
    clf = GaussianNB()
    clf.fit(X_train_lda.reshape(-1, 1), y_train)
    y_pred = clf.predict(X_test_lda.reshape(-1, 1))
    acc = accuracy_score(y_test, y_pred)
    print(f"Naive Bayes accuracy on test data (after LDA projection): {acc:.4f}")

    # Only plot ROC and compute AUC if binary classification
    if len(classes) == 2:
        # ROC and AUC before projection
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        print(f"AUC before projection: {roc_auc:.4f}")

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (before projection)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

        # ROC and AUC after LDA projection
        clf = GaussianNB()
        clf.fit(X_train_lda.reshape(-1, 1), y_train)
        y_score_lda = clf.predict_proba(X_test_lda.reshape(-1, 1))[:, 1]
        fpr_lda, tpr_lda, thresholds_lda = roc_curve(y_test, y_score_lda)
        roc_auc_lda = auc(fpr_lda, tpr_lda)
        print(f"AUC after LDA projection: {roc_auc_lda:.4f}")

        plt.figure()
        plt.plot(fpr_lda, tpr_lda, color='darkgreen', lw=2, label=f'ROC curve (AUC = {roc_auc_lda:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (after LDA projection)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()