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

target = "SpType-ELS"

excluded = {
    "Source", "Teff", "Dist", "Rad", "Lum-Flame",
    "Mass-Flame", "Age-Flame", "z-Flame",
}

features = [col for col in data_GaDR3.columns if col not in excluded and col != target]

data_GaDR3.head()
data_GaDR3.info()
data_GaDR3.isnull().sum().sort_values(ascending=False)
data_GaDR3 = data_GaDR3.drop(columns=excluded)

random_seed = NotImplemented
test_size = NotImplemented

GaussianNB().fit(data_GaDR3.drop(columns=[target]), data_GaDR3[target])


test_data = data_GaDR3.sample(frac=0.2, random_state=42)
y_test = GaussianNB().predict(test_data.drop(columns=[target]))   
train_data = data_GaDR3.drop(test_data.index)
GaussianNB().score(train_data.drop(columns=[target]), train_data[target])



X = data_GaDR3.drop(columns=['SpType-ELS', 'Source'])
y = data_GaDR3['SpType-ELS']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

#Print the sizes of the splits and the class distributions
print("Training size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))

print("Training classes:\n", y_train.value_counts())
print("Validation classes:\n", y_val.value_counts())
print("Test classes:\n", y_test.value_counts())

#---

# 