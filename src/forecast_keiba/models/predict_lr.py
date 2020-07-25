import pickle
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
import mlflow

mlflow.set_tracking_uri('../../../logs/mlruns/')
mlflow.set_experiment('forecast_keiba')

class ShutubaTable:
    # 出馬表を作る
    # = race plans
    def __init__(self):
        self.shutuba_table = pd.DataFrame()

    def scrape_shutuba_table(self, race_id_list):
        options = Options()
        options.binary_location = '/usr/bin/google-chrome'
        options.add_argument('--headless')
        options.add_argument('--window-size=1280,1024')
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome('chromedriver', chrome_options=options)
        for race_id in race_id_list:
            url  = 'https://race.netkeiba.com/race/shutuba.html?race_id='\
                      + race_id
            driver.get(url)
            elements = driver.find_elements_by_class_name('HorseList')
            for element in elements:
                row = []
                tds = element.find_elements_by_tag_name('td')
                for td in tds:
                    row.append(td.text)
                self.shutuba_table = self.shutuba_table.append(pd.Series(row, name=race_id))
        driver.close()


def preprocess_netkeiba_future(race_plans_df):
    df = race_plans_df.copy()

    df = df.rename(columns={0 :'枠番',1:'馬番',3:'馬名',4:'性齢',5:'斤量',6:'騎手',7:'厩舎',8:'馬体重',9:'単勝',10:'人気'})
    df['性'] = df['性齢'].map(lambda x:str(x)[0])
    df['年齢'] = df['性齢'].map(lambda x:str(x)[1:]).astype(int)
    df['体重'] = df['馬体重'].str.split('(',expand = True)[0].astype(int)
    df['体重変化'] = df['馬体重'].str.split('(',expand = True)[1].str[:-1].astype(int)
    df['単勝'] = df['単勝'].astype(float)
    df['人気'] = df['人気'].astype(float)
    df['枠番'] = df['枠番'].astype(int)
    df['馬番'] = df['馬番'].astype(int)
    df['斤量'] = df['斤量'].astype(float)
    df['斤量'] = df['斤量'].astype(int)
    df.drop([2,11,12,'厩舎','性齢','馬体重','馬名','騎手'],axis = 1,inplace = True)
    df['性_セ'] = [0] * 18
    df = pd.get_dummies(df)

    return df


def predict_lr(model_lr, parameters):
    predict_race_id = parameters["predict_race_id"]

    st = ShutubaTable()
    st.scrape_shutuba_table([predict_race_id])

    race_plans_df = st.shutuba_table
    race_plans_df_processed = preprocess_netkeiba_future(race_plans_df)

    # テストデータでの予測結果を取得し、出力する
    y_pred = model_lr.predict(race_plans_df_processed)
    print(pd.DataFrame({'pred':y_pred}))


def main(model_lr, parameters):
    return predict_lr(model_lr, parameters)

if __name__ == "__main__":
    main(model_lr, parameters)
