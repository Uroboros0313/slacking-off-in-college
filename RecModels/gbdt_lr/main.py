from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model import TreeLogit

DATA_DIR = Path('../data/user_data')

df = pd.read_csv(DATA_DIR / 'fm_data.csv')
y = df['rating']
del df['rating']
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

if __name__=='__main__':
    model = TreeLogit(n_estimators=20, tree_method='lgb')
    model.make_experiments(X_train, y_train, X_test, y_test)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    model.evaluate(y_test, pred)