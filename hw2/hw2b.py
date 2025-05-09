import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# # Fetch Breast Cancer dataset 
# breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
    
# # Get data (as pandas dataframes) 
# X = breast_cancer_wisconsin_diagnostic.data.features.values 
# y = breast_cancer_wisconsin_diagnostic.data.targets.values.ravel()

# fetch dataset 
# ionosphere = fetch_ucirepo(id=52) 

# # Get data (as pandas dataframes) 
# X = ionosphere.data.features.values 
# y = ionosphere.data.targets.values.ravel() 

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

# how many features?
# print("Number of features:", X.shape[1])

# Split into train/test (8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize data (fit only on training set)
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train_std = (X_train - X_mean) / X_std
X_test_std = (X_test - X_mean) / X_std

# Naive Bayes on original data
clf_orig = GaussianNB()
clf_orig.fit(X_train_std, y_train)
y_pred_orig = clf_orig.predict(X_test_std)
acc_orig = accuracy_score(y_test, y_pred_orig)
print(f"NB accuracy (original): {acc_orig:.4f}")

# PCA (fit only on training set)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
print("PCA components shape:", pca.components_.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Sum of explained variance ratio:", np.sum(pca.explained_variance_ratio_))

# Naive Bayes on PCA-reduced data
clf_pca = GaussianNB()
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)
print(f"NB accuracy (PCA): {acc_pca:.4f}")