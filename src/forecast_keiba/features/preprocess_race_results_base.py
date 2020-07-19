import pandas as pd

def preprocess_netkeiba_past(race_results_df: pd.DataFrame) -> pd.DataFrame:
    df = race_results_df.copy()

    # データ整形
    df = df[~(df['着順'].astype(str).str.contains('\D'))]
    df['着順'] = df['着順'].astype(int)
    df['性'] = df['性齢'].map(lambda x:str(x)[0])
    df['年齢'] = df['性齢'].map(lambda x:str(x)[1:]).astype(int)
    df['体重'] = df['馬体重'].str.split('(',expand = True)[0].astype(int)
    df['体重変化'] = df['馬体重'].str.split('(',expand = True)[1].str[:-1].astype(int)
    df['単勝'] = df['単勝'].astype(float)

    df.drop(['タイム','着差','調教師','性齢','馬体重'],axis = 1,inplace = True)

    # 4位より下はまとめる
    clip_rank = lambda x: x if x < 4 else 4
    df['rank'] = df['着順'].map(clip_rank)

    # test['馬名'].value_counts()などでカウントし、数が多そうなのは落とした後、ダミー変数化
    df.drop(['着順','馬名','騎手'], axis = 1,inplace = True)
    df = pd.get_dummies(df)

    return df

def preprocess_race_results_base(race_results_df):
    race_results_df_processed_base = preprocess_netkeiba_past(race_results_df)
    return race_results_df_processed_base

def main(race_results_df):
    return preprocess_race_results_base(race_results_df)

if __name__ == "__main__":
    main(race_results_df)