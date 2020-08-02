from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import lightgbm as lgb

def train_lightgbm(race_results_df_processed, parameters):
    print('train_lightgbm')
    # 説明変数の取得
    X = race_results_df_processed.drop(['rank'],axis=1)
    # 目的変数の取得
    y = race_results_df_processed['rank']

    # train と test に分離
    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.3,random_state=0)

    # ランダムアンダーサンプリング
    from imblearn.under_sampling import RandomUnderSampler

    cnt_rank_1 = y_train.value_counts()[1]
    cnt_rank_2 = y_train.value_counts()[2]
    cnt_rank_3 = y_train.value_counts()[3]

    rus = RandomUnderSampler(sampling_strategy={1:cnt_rank_1,2:cnt_rank_2,3:cnt_rank_3,4:cnt_rank_1},random_state=71)

    X_train_rus,y_train_rus = rus.fit_sample(X_train,y_train)

    # 学習
    params = {
        "num_leaves": 4,
        "n_estimators": 80,
        "class_weight": "balanced",
        "random_state": 100,
    }
    model_lightgbm = lgb.LGBMClassifier(**params)
    model_lightgbm.fit(X_train_rus.values,y_train_rus.values)
    print('train score: ' + str(model_lightgbm.score(X_train.values,y_train.values)))
    print('test score: ' + str(model_lightgbm.score(X_test.values,y_test.values)))

    # テストデータでの予測結果を取得し、出力する
    y_pred = model_lightgbm.predict(X_test.values)
    print(pd.DataFrame({'pred':y_pred,'actual':y_test.values}))

    return model_lightgbm

def main(race_results_df_processed, parameters):
    return train_lightgbm(race_results_df_processed, parameters)

if __name__ == "__main__":
    main(race_results_df_processed, parameters)
