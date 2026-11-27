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

