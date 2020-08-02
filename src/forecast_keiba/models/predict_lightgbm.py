import pandas as pd

def predict_lightgbm(model_lightgbm, race_results_df_processed_predict, parameters):
    print('start predict_lightgbm')

    # テストデータでの予測結果を取得し、出力する
    y_pred = model_lightgbm.predict(race_results_df_processed_predict.values)
    print(pd.DataFrame({'pred':y_pred}))


def main(model_lightgbm, race_results_df_processed_predict, parameters):
    return predict_lightgbm(model_lightgbm, race_results_df_processed_predict, parameters)

if __name__ == "__main__":
    main(model_lightgbm, race_results_df_processed_predict, parameters)
