import pandas as pd

def preprocess_netkeiba_future(race_plans_df, race_id_list):
    df = race_plans_df.copy()

    df = df.rename(columns={0 :'枠番',1:'馬番',3:'馬名',4:'性齢',5:'斤量',6:'騎手',7:'厩舎',8:'馬体重',9:'単勝',10:'人気'})
    df['性'] = df['性齢'].map(lambda x:str(x)[0])
    df['年齢'] = df['性齢'].map(lambda x:str(x)[1:]).astype(int)
    df['体重'] = df['馬体重'].str.split('(',expand = True)[0].astype(int)
    df['体重変化'] = df['馬体重'].str.split('(',expand = True)[1].str[:-1].astype(int)
    df['体重変化'] = [int(s) for s in list(df['体重変化'])]
    df['単勝'] = df['単勝'].astype(float)
    df['人気'] = df['人気'].astype(float)
    df['枠番'] = df['枠番'].astype(int)
    df['馬番'] = df['馬番'].astype(int)
    df['斤量'] = df['斤量'].astype(float)
    df['斤量'] = df['斤量'].astype(int)
    df['course_id'] = [int(race_id_list[0][4:6])]*len(df)
    df.drop([2,11,12,'厩舎','性齢','馬体重','馬名','騎手','horse_id','jockey_id','date'],axis = 1,inplace = True)
    df = pd.get_dummies(df)

    return df


def preprocess_race_results_predict(race_results_df_predict, race_results_df_processed, parameters):
    print('start preprocess_race_results_predict')

    predict_race_id = parameters['predict_race_id']
    race_plans_df_processed = preprocess_netkeiba_future(race_results_df_predict, [predict_race_id])
    # カラムを合わせる
    race_plans_df_processed = pd.get_dummies(pd.concat([race_results_df_processed, race_plans_df_processed]))
    race_plans_df_processed = race_plans_df_processed[race_results_df_processed.columns]
    race_plans_df_processed = race_plans_df_processed.drop(['rank'],axis=1)
    # 予想対象レース抽出
    race_plans_df_processed = race_plans_df_processed.query('index==@predict_race_id')
    race_plans_df_processed = race_plans_df_processed.fillna(0)
    race_results_df_processed_predict = race_plans_df_processed

    return race_results_df_processed_predict

def main(race_results_df_predict, race_results_df_processed, parameters):
    return preprocess_race_results_predict(race_results_df_predict, race_results_df_processed, parameters)

if __name__ == "__main__":
    main(race_results_df_predict, race_results_df_processed, parameters)