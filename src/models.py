import numpy as np
import optuna
import matplotlib.pyplot as plt
import pandas as pd
import lime
from lime import lime_tabular
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder



data_GaDR3 = pd.read_csv('data/dataGaia.csv')

data_GaDR3.head()
data_GaDR3.info()
data_GaDR3.isnull().sum().sort_values(ascending=False)
data_GaDR3 = data_GaDR3.drop(columns=["Unnamed: 0", "Teff"])



# 3 way split of the data into training, validation, and test sets (70-15-15)

X = data_GaDR3.drop(columns=['SpType-ELS', 'Source'])
y = data_GaDR3['SpType-ELS']

# 70%
X_train, X_temp, y_train_raw, y_temp_raw = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 30% then split into 15% validation and 15% test
X_val, X_test, y_val_raw, y_test_raw = train_test_split(
    X_temp, y_temp_raw, test_size=0.50, random_state=42, stratify=y_temp_raw
)

#Print the sizes of the splits and the class distributions
print("Train size:", len(X_train))
print("Val size:", len(X_val))
print("Test size:", len(X_test))

print("Train classes:\n", y_train.value_counts())
print("Val classes:\n", y_val.value_counts())
print("Test classes:\n", y_test.value_counts())