import pandas as pd
import time
from tqdm.notebook import tqdm
import datetime
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

## 前処理
def preprocessing(results):
    df = results.copy()
    
    df = df[~(df['着順'].astype(str).str.contains('\D'))]
    df['着順'] = df['着順'].astype(int)
    df['性'] = df['性齢'].map(lambda x:str(x)[0])
    df['所属'] = df['調教師'].map(lambda x:str(x)[1])
    df['年齢'] = df['性齢'].map(lambda x:str(x)[1:]).astype(int)
    df['体重'] = df['馬体重'].str.split('(',expand = True)[0].astype(int)
    df['体重変化'] = df['馬体重'].str.split('(',expand = True)[1].str[:-1]
    
    object_to_int = [int(s) for s in list(df['体重変化'])]
    df['体重変化'] = object_to_int
    
    le = LabelEncoder()
    le = le.fit(df['Borned_place'])
    df['Borned_place'] = le.transform(df['Borned_place'])
    
    df.drop(['タイム'],axis=1,inplace=True)
    df.drop(['着差'],axis=1,inplace=True)
    df.drop(['調教師'],axis=1,inplace=True)
    df.drop(['性齢'],axis=1,inplace=True)
    df.drop(['馬体重'],axis=1,inplace=True)
    df.drop(['馬名'],axis=1,inplace=True)
    df.drop(['騎手'],axis=1,inplace=True)
    df.drop(['単勝'],axis=1,inplace=True)
    df.drop(['人気'],axis=1,inplace=True)
    df.drop(['horse_id'],axis=1,inplace=True)
    df.drop(['jockey_id'],axis=1,inplace=True)
    return df

# カテゴリー化とダミー化
def process_categorical(df, target_columns):
    df2 = df.copy()
    for column in target_columns:
        df2[column] = LabelEncoder().fit_transform(df2[column].fillna('Na'))
    
    #target_columns以外にカテゴリ変数があれば、ダミー変数にする
    df2 = pd.get_dummies(df2)

    for column in target_columns:
        df2[column] = df2[column].astype('category')
        
    df2 = df2.fillna(0)

    return df2

# メイン関数
def preprocess_race_results(race_results_df, parameters):
    print('start preprocess_race_results')
    race_results_df_processed = preprocessing(race_results_df)
    target_columns = []
    for i in range(62):
        target_columns.append('peds_'+str(i))
    race_results_df_processed = process_categorical(race_results_df_processed, target_columns)
    race_results_df_processed.to_pickle('../../../data/processed/race_results_df_processed.pickle')
    return race_results_df_processed

def main(race_results_df, parameters):
    return preprocess_race_results(race_results_df, parameters)

if __name__ == "__main__":
    main(race_results_df, parameters)