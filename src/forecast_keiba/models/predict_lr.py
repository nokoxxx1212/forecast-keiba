import pandas as pd

def predict_lr(model_lr, race_results_df_processed_predict, parameters):
    print('start predict_lr')

    # テストデータでの予測結果を取得し、出力する
    y_pred = model_lr.predict(race_results_df_processed_predict)
    print(pd.DataFrame({'pred':y_pred}))


def main(model_lr, race_results_df_processed_predict, parameters):
    return predict_lr(model_lr, race_results_df_processed_predict, parameters)

if __name__ == "__main__":
    main(model_lr, race_results_df_processed_predict, parameters)
