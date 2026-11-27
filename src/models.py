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

