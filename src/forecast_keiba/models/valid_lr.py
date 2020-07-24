import pandas as pd
import random
import mlflow
import os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def valid_lr(race_results_df_processed_valid, model_lr):
    # mlflow
    print('FILE_DIR: ' + FILE_DIR)
    mlflow.set_tracking_uri(FILE_DIR + '/../../../logs/mlruns/')
    mlflow.set_experiment('forecast_keiba_valid')
    mlflow.start_run()

    # 検証のデータ準備
    race_results_df_processed_valid = race_results_df_processed_valid
    # 説明変数の取得
    X_valid = race_results_df_processed_valid.drop(['rank'],axis=1)
    # 目的変数の取得
    y_valid = race_results_df_processed_valid['rank']

    # 推論実行
    y_valid_pred = model_lr.predict(X_valid)

    # 集計用に処理
    valid_results_df = pd.DataFrame({'pred':y_valid_pred,'actual':y_valid})
    race_id_list = list(set(list(valid_results_df.index)))
    valid_results_list = valid_results_df.reset_index().values.tolist()
    # シャッフル
    random.shuffle(valid_results_list)

    # 集計（馬単）
    correct_count = 0
    for race_id in race_id_list:
        pred_cnt_by_race = 0
        cnt_by_race = 0
        for rank in [1]:
            for i in range(len(valid_results_list)):
                # 対象レースidのうち、{rank}位と予測された馬
                if valid_results_list[i][0] == race_id and valid_results_list[i][1] == rank:
                    pred_cnt_by_race += 1
                    if pred_cnt_by_race <= 1 and (valid_results_list[i][2] == 1):
                        cnt_by_race += 1
        if cnt_by_race == 1:
            correct_count += 1
    acc_exacta_1 = correct_count/100
    print('acc_exacta_1: ' + str(acc_exacta_1))

    # 集計（馬連）
    correct_count = 0
    for race_id in race_id_list:
        pred_cnt_by_race = 0
        cnt_by_race = 0
        for rank in [1, 2]:
            for i in range(len(valid_results_list)):
                # 対象レースidのうち、{rank}位と予測された馬
                if valid_results_list[i][0] == race_id and valid_results_list[i][1] == rank:
                    pred_cnt_by_race += 1
                    if pred_cnt_by_race <= 2 and (valid_results_list[i][2] == 1 or valid_results_list[i][2] == 2):
                        cnt_by_race += 1
        if cnt_by_race == 2:
            correct_count += 1
    acc_quinella_2 = correct_count/100
    print('acc_quinella_2: ' + str(acc_quinella_2))

    # 集計（三連複）
    correct_count = 0
    for race_id in race_id_list:
        pred_cnt_by_race = 0
        cnt_by_race = 0
        for rank in [1, 2, 3]:
            for i in range(len(valid_results_list)):
                # 対象レースidのうち、{rank}位と予測された馬
                if valid_results_list[i][0] == race_id and valid_results_list[i][1] == rank:
                    pred_cnt_by_race += 1
                    if pred_cnt_by_race <= 3 and (valid_results_list[i][2] == 1 or valid_results_list[i][2] == 2 or valid_results_list[i][2] == 3):
                        cnt_by_race += 1
        if cnt_by_race == 3:
            correct_count += 1
    acc_trio_3 = correct_count/100
    print('acc_trio_3: ' + str(acc_trio_3))

    mlflow.log_metric("acc_exacta_1", acc_exacta_1)
    mlflow.log_metric("acc_quinella_2", acc_quinella_2)
    mlflow.log_metric("acc_trio_3", acc_trio_3)
    mlflow.end_run()


def main(race_results_df_processed_valid, model_lr):
    return valid_lr(race_results_df_processed_valid, model_lr)

if __name__ == "__main__":
    main(race_results_df_processed_valid, model_lr)