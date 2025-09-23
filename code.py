import pandas as pd
import numpy as np
from utils import *


df = pd.read_csv('data/breast-cancer.csv')

df.drop('id', axis=1, inplace=True)
df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)

X = df.drop(columns=['diagnosis']).values
y = df['diagnosis'].values

X_train, X_test, y_train, y_test = train_test_split(X, y)