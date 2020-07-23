import pandas as pd

def valid_lr(race_results_df_processed_valid, model_lr):
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

    # 集計（1位正解率）
    correct_count = 0
    for race_id in race_id_list:
        pred_1_cnt_by_race = 0
        for i in range(len(valid_results_list)):
            # 対象レースidのうち、一位と予測された馬
            if valid_results_list[i][0] == race_id and valid_results_list[i][1] == 1:
                pred_1_cnt_by_race += 1
                # 対象レースidのうち一位と予測された馬が一つ目で、かつ結果も1位の場合
                if pred_1_cnt_by_race == 1 and valid_results_list[i][2] == 1:
                    correct_count += 1
    print('rank1_acc: ' + str(correct_count/100))

    # 集計（1-3位正解率）
    correct_count = 0
    for race_id in race_id_list:
        pred_3_cnt_by_race = 0
        for rank in [1, 2, 3]:
            for i in range(len(valid_results_list)):
                # 対象レースidのうち、{rank}位と予測された馬
                if valid_results_list[i][0] == race_id and valid_results_list[i][1] == rank:
                    pred_3_cnt_by_race += 1
                    # 対象レースidのうち一位と予測された馬が一つ目で、かつ結果も1位の場合
                    if pred_3_cnt_by_race <= 3 and valid_results_list[i][2] == 1 or valid_results_list[i][2] == 2 or valid_results_list[i][2] == 3:
                        correct_count += 1
    print('rank3_acc: ' + str(correct_count/300))


def main(race_results_df_processed_valid, model_lr):
    return valid_lr(race_results_df_processed_valid, model_lr)

if __name__ == "__main__":
    main(race_results_df_processed_valid, model_lr)