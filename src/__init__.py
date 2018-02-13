from data.make_dataset import *
from visualization.visualize import *
from models.predictions import *
from models.modelling import *
import pandas as pd

df_train = pd.read_csv('../data/raw/train_values.csv', parse_dates=['date_recorded'])
df_labels = pd.read_csv('../data/raw/train_labels.csv')
df_test = pd.read_csv('../data/raw/test_values.csv', parse_dates=['date_recorded'])

visualize(df_train, df_labels)

X_train, X_test = createArray(df_train, df_labels, df_test)

best_model = model_selection(X_train, X_test, df_labels)

prediction_making(best_model, X_test)
