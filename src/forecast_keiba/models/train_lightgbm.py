import pandas as pd
import pickle
import time
from tqdm.notebook import tqdm
import datetime as dt
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb

def split_data(df,test_size):
    sorted_id_list = df.sort_values('date').index.unique()
    train_id_list = sorted_id_list[:round(len(sorted_id_list)*(1-test_size))]
    test_id_list = sorted_id_list[round(len(sorted_id_list)*(1-test_size)):]
    train = df.loc[train_id_list]
    test = df.loc[test_id_list]
    return train,test

# メイン関数
def train_lightgbm(race_results_df_processed, parameters):
    print('start train_lr')
    race_results_df_processed.sort_values('date', ascending=False)
    df = race_results_df_processed[(race_results_df_processed['date'] > dt.datetime(2010,1,1)) & (race_results_df_processed['date'] < dt.datetime(2019,12,31))]
    df['rank'] = df['着順'].map(lambda x: 1 if x < 6 else (3 if x > 11 else 2))
    df.to_pickle('../../../data/processed/train_data.pickle')
    train,test = split_data(df,0.3)
    rank_1 = train['rank'].value_counts()[1]
    rank_2 = train['rank'].value_counts()[2]
    rank_3 = train['rank'].value_counts()[3]
    rus = RandomUnderSampler(sampling_strategy={1:rank_1,2:rank_2,3:rank_3},random_state=71)

    X_train = train.drop(['着順','date','rank'],axis=1)
    y_train = train['rank']
    X_test = test.drop(['着順','date','rank'],axis=1)
    y_test = test['rank']

    X_train_rus,y_train_rus = rus.fit_sample(X_train,y_train)
    
    params = {
    "num_leaves": 64,
    "n_estimators": 80,
    "class_weight": "balanced",
    "random_state": 100,
    "max_depth": 24,
    }

    model_lightgbm = lgb.LGBMClassifier(**params)
    model_lightgbm.fit(X_train_rus.values,y_train_rus.values)

    print('train score: ' + str(model_lightgbm.score(X_train,y_train)))
    print('test score: ' + str(model_lightgbm.score(X_test,y_test)))
    
    print('feature importances')
    importances = pd.DataFrame(
    {"features": X_train.columns, "importance": model_lightgbm.feature_importances_}
    )
    print(importances.sort_values("importance", ascending=False)[:20])
    pickle.dump(model_lightgbm, open('../../../data/models/model_lightgbm', 'wb'))
    
    return model_lightgbm