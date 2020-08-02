import pandas as pd

def predict_rf(model_rf, race_results_df_processed_predict, parameters):
    print('start predict_rf')

    # テストデータでの予測結果を取得し、出力する
    y_pred = model_rf.predict(race_results_df_processed_predict)
    print(pd.DataFrame({'pred':y_pred}))


def main(model_rf, race_results_df_processed_predict, parameters):
    return predict_rf(model_rf, race_results_df_processed_predict, parameters)

if __name__ == "__main__":
    main(model_rf, race_results_df_processed_predict, parameters)
