import pandas as pd
import time
from tqdm.notebook import tqdm
import datetime
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
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
from selenium.webdriver import Chrome, ChromeOptions

# レース詳細情報取得
def scrape_race_info(soup, parameters):
    text_race_data = str(soup.find('div',attrs={'class':'RaceData01'}))
    race_data = soup.find('div',attrs={'class':'RaceData01'})
        
    whether_text = [text_race_data[text_race_data.find("天候")+3:text_race_data.find('<span class="Icon_Weather')]]
    course_type_text = [text_race_data[text_race_data.find("(")+1:text_race_data.find(")")]]
    ground_type_text = [race_data.find_all('span')[0].text]
    ground_state_text = [race_data.find_all('span')[2].text[race_data.find_all('span')[2].text.find(":")+1:]]

    # race_info = ground_state_text+ ground_type_text + whether_text + course_type_text + ['2020年9月6日']
    race_info = ground_state_text+ ground_type_text + whether_text + course_type_text + parameters['predict_date']
    
    info_dict = {}

    title = soup.find('title').text
    if "G1" in title:
        info_dict["Grade"] = "1"
    elif "G2" in title:
        info_dict["Grade"] = "2"
    elif "G3" in title:
        info_dict["Grade"] = "3"
    else:
        info_dict["Grade"] = "4"

    for text in race_info:
        if "芝" in text:
            info_dict["race_type"] = '芝'
        if "ダ" in text:
            info_dict["race_type"] = 'ダート'
        if "障" in text:
            info_dict["race_type"] = "障害"
        if "m" in text:
            info_dict["course_len"] = int(re.findall(r"\d+", text)[0])
        if text in ["良","稍重", "重", "不良"]:
            info_dict["ground_state"] = text
        if text in "稍":
            info_dict["ground_state"] = "稍重"
        if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
            info_dict["weather"] = text
        if "年" in text:
            info_dict["date"] = text
        if "右" in text:
            info_dict["course_type"] = "right"
        if "左" in text:
            info_dict["course_type"] = "left"
        if "直線" in text:
            info_dict["course_type"] = "straight"

    return info_dict

# 馬とジョッキーのID取得
def scrape_id(soup,id_name):
    id_list = []

    if id_name == "horse":
        words  = soup.find_all("td", attrs={"class": "HorseInfo"})
        for word in words:
            id_list.append(word.find("a").get('href')[-10:])
    elif id_name == "jockey":
        words  = soup.find_all("td", attrs={"class": "Jockey"})
        for word in words:
            id_list.append(word.find("a").get('href')[-6:-1])
        
    return id_list

# 出馬表作成
def make_horse_table(df):
    df_tmp = pd.DataFrame()
    df_tmp["枠番"] = df[("枠","枠")].values
    df_tmp["馬番"] = df[("馬番","馬番")].values
    df_tmp["馬名"] = df[("馬名","馬名")].values
    df_tmp["性齢"] = df[("性齢","性齢")].values
    df_tmp["斤量"] = df[("斤量","斤量")].values
    df_tmp["騎手"] = df[("騎手","騎手")].values
    df_tmp["厩舎"] = df[("厩舎","厩舎")].values
    df_tmp["馬体重(増減)"] = df[("馬体重(増減)","馬体重(増減)")].values

    return df_tmp

# 出走間隔取得
def scrape_race_span(race_id,df_length):
    url = "https://race.netkeiba.com/race/shutuba_past.html?race_id="+race_id+"&rf=shutuba_submenu"
    df = pd.read_html(url)[0]
    texts = list(df["馬名"].values)
    race_span = []
    for text in texts:
        words = text.split( )
        flag = False
        for word in words:
            if "中" in word and "週" in word:
                race_span.append(int(word[1:-1]))
                flag = True
                continue
            elif "連闘" in word:
                race_span.append(1)
                flag = True
                continue
        if flag == False:
            race_span.append(0)
    if len(df) != df_length:
        flag = False
    else:
        flag = True
    time.sleep(0.1)
    return race_span,flag

# レース情報取得
def scrape_race_predict(race_id_list, parameters):
    race_tables = {}
    race_infos = {}
    for race_id in race_id_list:
        url = "https://race.netkeiba.com/race/shutuba.html?race_id=" + race_id + "&rf=race_submenu"
        df = pd.read_html(url)[0]
        table = make_horse_table(df)

        html = requests.get(url)
        html.encoding = "EUC-JP"
        soup = BeautifulSoup(html.text, "html.parser")

        race_span,flag = scrape_race_span(race_id,len(df))
        if flag != True:
            print("length error",race_id)
            continue
        table["race_span"] = race_span

        horse_id_list = scrape_id(soup,"horse")
        jockey_id_list = scrape_id(soup,"jockey")

        if len(horse_id_list) != len(df) or len(jockey_id_list) != len(df):
            continue

        table["horse_id"] = horse_id_list
        table["jockey_id"] = jockey_id_list
        table['course_id'] = [int(race_id[4:6])]*len(df)

        info_dict = {}
        info_dict = scrape_race_info(soup, parameters)
        race_tables[race_id] = table
        race_infos[race_id] = info_dict

    return race_tables,race_infos

# 馬の戦績取得
def scrape_horse_results(horse_id_list, pre_horse_id=[]):
    horse_results = {}
    for horse_id in tqdm(horse_id_list):
        if horse_id in pre_horse_id:
            continue
        try:
            url = 'https://db.netkeiba.com/horse/' + horse_id
            
            html = requests.get(url)
            html.encoding = "EUC-JP"
            soup = BeautifulSoup(html.text, "html.parser")
            
            texts = soup.find("div", attrs={"class": "db_prof_area_02"}).find_all("a")
            for text in texts:
                if "breeder" in str(text):
                    Borned_place = str(text)[str(text).find('e="')+3:str(text).find('">')]
            
            df = pd.read_html(url)[3]
            if df.columns[0]=='受賞歴':
                df = pd.read_html(url)[4]

            df["Borned_place"] = Borned_place

            horse_results[horse_id] = df
            
            time.sleep(0.1)
        except IndexError:
            continue
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
            break
        except:
            break
    return horse_results

# 馬の詳細戦績取得
class HorseResults:
    def __init__(self, horse_results):
        self.horse_results = horse_results[['日付','着順', '賞金']]
        self.preprocessing()

    def preprocessing(self):
        df = self.horse_results.copy()

        # 着順に数字以外の文字列が含まれているものを取り除く
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        df.dropna(subset=['着順'], inplace=True)
        df['着順'] = df['着順'].astype(int)
        df['着順'].fillna(0, inplace=True)

        df["date"] = pd.to_datetime(df["日付"])
        #df.drop(['日付'], axis=1, inplace=True)

        #賞金のNaNを0で埋める
        df['賞金'].fillna(0, inplace=True)

        self.horse_results = df

    def average(self, horse_id_list, date, n_samples='all'):
        self.horse_results.reindex(horse_id_list, axis=1)
        target_df = self.horse_results.loc[horse_id_list]

        #過去何走分取り出すか指定
        if n_samples == 'all':
            filtered_df = target_df[target_df['date'] < date]
        elif n_samples > 0:
            filtered_df = target_df[target_df['date'] < date].sort_values('date', ascending=False).groupby(level=0).head(n_samples)
        else:
            raise Exception('n_samples must be >0')

        average = filtered_df.groupby(level=0)[['着順', '賞金']].mean()
        return average.rename(columns={'着順':'着順_{}R'.format(n_samples), '賞金':'賞金_{}R'.format(n_samples)})
    # change 馬の最高賞金追加
    def max_money(self, horse_id_list, date, n_samples='all'):
        self.horse_results.reindex(horse_id_list, axis=1)
        target_df = self.horse_results.loc[horse_id_list]
        
        #過去何走分取り出すか指定
        if n_samples == 'all':
            filtered_df = target_df[target_df['date'] < date]
        elif n_samples > 0:
            filtered_df = target_df[target_df['date'] < date].sort_values('date', ascending=False).groupby(level=0).head(n_samples)
        else:
            raise Exception('n_samples must be >0')
            
        max_money = filtered_df.groupby(level=0)[['賞金']].max()
        return max_money.rename(columns={'賞金':'最高賞金_{}R'.format(n_samples)})

    def merge(self, results, date, n_samples='all'):
        df = results[results['date']==date]
        horse_id_list = df['horse_id']
        merged_df = df.merge(self.average(horse_id_list, date, 3), left_on='horse_id',right_index=True, how='left')\
                      .merge(self.average(horse_id_list, date, 5), left_on='horse_id',right_index=True, how='left')\
                      .merge(self.average(horse_id_list, date, "all"), left_on='horse_id',right_index=True, how='left')\
                      .merge(self.max_money(horse_id_list, date, 'all'), left_on='horse_id',right_index=True, how='left')
        return merged_df

    def merge_all(self, results, n_samples='all'):
        date_list = results['date'].unique()
        merged_df = pd.concat([self.merge(results, date, n_samples) for date in tqdm(date_list)])
        return merged_df

# 血統データ取得
def scrape_peds(horse_id_list, pre_peds={}):
    peds = pre_peds
    for horse_id in tqdm(horse_id_list):
        if horse_id in peds.keys():
            continue
        try:
            url = "https://db.netkeiba.com/horse/ped/" + horse_id
            df = pd.read_html(url)[0]

            generations = {}
            for i in reversed(range(5)):
                generations[i] = df[i]
                df.drop([i], axis=1, inplace=True)
                df = df.drop_duplicates()

            ped = pd.concat([generations[i] for i in range(5)]).rename(horse_id)
            peds[horse_id] = ped.reset_index(drop=True)
            time.sleep(0.1)
        except IndexError:
            continue
        except Exception as e:
            print(e)
            break
    return peds

# カテゴリー化およびダミー化
def process_categorical(df, target_columns):
    df2 = df.copy()
    for column in target_columns:
        df2[column] = LabelEncoder().fit_transform(df2[column].fillna('Na'))
    
    #target_columns以外にカテゴリ変数があれば、ダミー変数にする
    df2 = pd.get_dummies(df2)

    for column in target_columns:
        df2[column] = df2[column].astype('category')

    return df2

# 血統データ結合
def add_blood_data(horse_id_list,df):
    peds = scrape_peds(horse_id_list)
    peds = pd.concat([peds[horse_id] for horse_id in peds], axis=1).T
    peds = peds.add_prefix('peds_')
    df = df.merge(peds,left_on='horse_id', right_index=True, how='left')
    return df

# ジョッキー情報取得
def scrape_jockey_results(jockey_id_list, pre_jockey_id=[]):
    jockey_results = {}
    for jockey_id in tqdm(jockey_id_list):
        if jockey_id in pre_jockey_id:
            continue
        try:
            url = 'https://db.netkeiba.com/jockey/result/' + jockey_id + '/'
            df = pd.read_html(url)[0][['勝率','連対率','複勝率']][:1]
            jockey_results[jockey_id] = df
            time.sleep(0.1)
        except IndexError:
            continue
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
            break
        except:
            break
    return jockey_results

# 前処理
def preprocessing_predict(df):
    df['性'] = df['性齢'].map(lambda x:str(x)[0])
    df['年齢'] = df['性齢'].map(lambda x:str(x)[1:]).astype(int)
    df['体重'] = df['馬体重(増減)'].str.split('(',expand = True)[0].astype(int)
    df['体重変化'] = df['馬体重(増減)'].str.split('(',expand = True)[1].str[:-1]
    df.loc[df['体重変化'] == "前計不", '体重変化'] = 0
    object_to_int = [int(s) for s in list(df['体重変化'])]
    df['体重変化'] = object_to_int
    df['枠番'] = df['枠番'].astype(int)
    df['馬番'] = df['馬番'].astype(int)
    df['斤量'] = df['斤量'].astype(float)
    df['斤量'] = df['斤量'].astype(int)
    df['所属'] = df['厩舎'].map(lambda x:str(x)[:2])
    df = df.replace('栗東', '西')
    df = df.replace('美浦', '東')
    df = df.replace('地方', '地')
    df = df.replace('海外', '海')
    
    horse_name = df["馬名"]
    
    df.drop(['性齢','馬体重(増減)'],axis = 1,inplace = True)
    df.drop(['馬名'],axis=1,inplace=True)
    df.drop(['騎手'],axis=1,inplace=True)
    df.drop(['厩舎'],axis=1,inplace=True)
    df.drop(['horse_id'],axis=1,inplace=True)
    df.drop(['jockey_id'],axis=1,inplace=True)
    df.drop(['date'],axis=1,inplace=True)
    
    le = LabelEncoder()
    le = le.fit(df['Borned_place'])
    df['Borned_place'] = le.transform(df['Borned_place'])

    return df.fillna(0),horse_name

# 前処理とダミー化
def preprocess_race_predict(df):
    preprocess_df,horse_name = preprocessing_predict(df)
    target_columns = []
    for i in range(62):
        target_columns.append('peds_'+str(i))
    preprocess_df = process_categorical(preprocess_df, target_columns)
    return preprocess_df,horse_name

# 学習と推論でデータに差がないか比較
def compare_df(df):
    df_predict = df.copy()

    df_train = pd.read_pickle('../../../data/processed/train_data.pickle')
    df_train.drop(['date'],axis=1,inplace=True)
    df_train.drop(['着順'],axis=1,inplace=True)
    df_train.drop(['rank'],axis=1,inplace=True)
    horse_count = len(df_predict)
    train_colums_list = list(df_train.columns.values)
    predict_colums_list = list(df_predict.columns.values)
    for t_column in train_colums_list:
        if t_column not in predict_colums_list:
            df_predict[t_column] = [0]*horse_count
            
    return df_predict.fillna(0).reindex(columns=train_colums_list)


# メイン関数
def predict_lightgbm(model_lightgbm, parameters):

    print("レース結果取得中")
    race_tables,race_infos = scrape_race_predict(parameters['predict_race_id'], parameters)
    for key in race_tables:
        race_tables[key].index = [key] * len(race_tables[key])
    race_tables = pd.concat([race_tables[key] for key in race_tables], sort=False)
    df_infos = pd.DataFrame(race_infos.values(), index=race_infos.keys())
    predict_addinfo = race_tables.merge(df_infos,left_index=True,right_index=True,how='inner')
    predict_addinfo['date'] = pd.to_datetime(predict_addinfo['date'],format='%Y年%m月%d日')

    print("馬情報取得中")
    horse_id_list = predict_addinfo['horse_id'].unique()
    horse_results = scrape_horse_results(horse_id_list)
    for key in horse_results:
        horse_results[key].index = [key] * len(horse_results[key])
    df_horse_results = pd.concat([horse_results[key] for key in horse_results])
    
    print("ジョッキー情報取得中")
    jockey_id_list = predict_addinfo['jockey_id'].unique()
    jockey_results = scrape_jockey_results(jockey_id_list)
    for key in jockey_results:
        jockey_results[key].index = [key] * len(jockey_results[key])
    df_jockey_results = pd.concat([jockey_results[key] for key in jockey_results])
    predict_addinfo = predict_addinfo.merge(df_jockey_results,left_on='jockey_id',right_index=True,how='left')

    print("馬の生産地取得")
    borned_place_list = []
    for i in range(len(predict_addinfo)):
        borned_place_list.append(list(set(list(horse_results[predict_addinfo['horse_id'][i]]["Borned_place"])))[0])
    predict_addinfo["Borned_place"] = borned_place_list

    print("データ結合中")
    hr = HorseResults(df_horse_results)
    predict_all = hr.merge_all(predict_addinfo, n_samples=5)
        
    print("血統情報取得中")
    add_blood_predict = add_blood_data(horse_id_list,predict_all)
    preprocess_df,horse_name = preprocess_race_predict(add_blood_predict)
    predict_data = compare_df(preprocess_df)
    
    return predict_data,horse_name

def main(model_lightgbm, parameters):
    return predict_lightgbm(model_lightgbm, race_results_df_processed_predict, parameters)

if __name__ == "__main__":
    main(model_lightgbm, parameters)
