# URLの母数を取得して、RACR_URL_DIR配下にテキストで保存する
# -> RACR_URL_DIR配下のテキストのレースIDを母数に、レース結果をスクレイピングする
# race_url/YYYY-MM.txt, race_results_df.pickle を事前に配置しておくと、ダウンロード量が減る

import datetime
import pytz
import re
import time
import os
from os import path
from selenium import webdriver
from selenium.webdriver.support.ui import Select,WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
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

# レースID取得
def get_race_url_by_year_and_mon(driver, year, month):
    URL = "https://db.netkeiba.com/?pid=race_search_detail"
    RACR_URL_DIR = '../../../data/raw/race_url'
    race_url_file = RACR_URL_DIR + "/" + str(year) + "-" + str(month) + ".txt" #保存先ファイル
    if not os.path.exists(RACR_URL_DIR):
        os.makedirs(RACR_URL_DIR)

    # URLにアクセス
    wait = WebDriverWait(driver,10)
    driver.get(URL)
    time.sleep(1)
    wait.until(EC.presence_of_all_elements_located)

    print('get_race_url_by_year_and_mon year: ' + str(year) + ' month: ' + str(month))

    # 期間を選択
    start_year_element = driver.find_element_by_name('start_year')
    start_year_select = Select(start_year_element)
    start_year_select.select_by_value(str(year))
    start_mon_element = driver.find_element_by_name('start_mon')
    start_mon_select = Select(start_mon_element)
    start_mon_select.select_by_value(str(month))
    end_year_element = driver.find_element_by_name('end_year')
    end_year_select = Select(end_year_element)
    end_year_select.select_by_value(str(year))
    end_mon_element = driver.find_element_by_name('end_mon')
    end_mon_select = Select(end_mon_element)
    end_mon_select.select_by_value(str(month))

    # 競馬場をチェック
    for i in range(1,11):
        terms = driver.find_element_by_id("check_Jyo_"+ str(i).zfill(2))
        terms.click()

    # 表示件数を選択(20,50,100の中から最大の100へ)
    list_element = driver.find_element_by_name('list')
    list_select = Select(list_element)
    list_select.select_by_value("100")

    # フォームを送信
    frm = driver.find_element_by_css_selector("#db_search_detail_form > form")
    frm.submit()
    time.sleep(1)
    wait.until(EC.presence_of_all_elements_located)

    total_num_and_now_num = driver.find_element_by_xpath("//*[@id='contents_liquid']/div[1]/div[2]").text
    total_num = int(re.search(r'(.*)件中', total_num_and_now_num).group().strip("件中"))

    pre_url_num = 0
    if os.path.isfile(race_url_file):
        with open(race_url_file, mode='r') as f:
            pre_url_num = len(f.readlines())

    if total_num!=pre_url_num:
        with open(race_url_file, mode='w') as f:
            #tableからraceのURLを取得(ループしてページ送り)
            total = 0
            while True:
                time.sleep(1)
                wait.until(EC.presence_of_all_elements_located)

                all_rows = driver.find_element_by_class_name('race_table_01').find_elements_by_tag_name("tr")
                total += len(all_rows)-1
                for row in range(1, len(all_rows)):
                    race_href = all_rows[row].find_elements_by_tag_name("td")[4].find_element_by_tag_name("a").get_attribute("href")
                    f.write(race_href+"\n")
                try:
                    target = driver.find_elements_by_link_text("次")[0]
                    driver.execute_script("arguments[0].click();", target) #javascriptでクリック処理
                except IndexError:
                    break
        print("got "+ str(total) +" urls of " + str(total_num) +" ("+str(year) +" "+ str(month) + ")")
    else:
        print("already have " + str(pre_url_num) +" urls ("+str(year) +" "+ str(month) + ")")

def get_race_url(parameters):
    RACR_URL_DIR = '../../../data/raw/race_url'
    now_datetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    options = Options()
    options.binary_location = '/usr/bin/google-chrome'
    options.add_argument('--headless')
    options.add_argument('--window-size=1280,1024')
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome('chromedriver', chrome_options=options) # 去年までのデータ
    for year in range(parameters["scrape_year_start"], parameters["scrape_year_end"]+1):
        for month in range(1, 13):
            race_url_file = RACR_URL_DIR + "/" + str(year) + "-" + str(month) + ".txt" #保存先ファイル
            if not os.path.isfile(race_url_file): # まだ取得していなければ取得
                get_race_url_by_year_and_mon(driver,year,month)
    # 先月までのデータ
    for year in range(now_datetime.year, now_datetime.year+1):
        for month in range(1, now_datetime.month):
            race_url_file = RACR_URL_DIR + "/" + str(year) + "-" + str(month) + ".txt" #保存先ファイル
            if not os.path.isfile(race_url_file): # まだ取得していなければ取得
                get_race_url_by_year_and_mon(driver,year,month)
    # 今月分は毎回取得
    get_race_url_by_year_and_mon(driver, now_datetime.year, now_datetime.month)

    driver.close()
    driver.quit()

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

# 馬とジョッキーのID取得
def scrape_id(soup,id_name):
    id_list = []
    words = soup.find("table", attrs={"summary": "レース結果"}).find_all(
        "a", attrs={"href": re.compile("^/"+id_name)}
    )

    for word in words:
        id = re.findall(r"\d+", word["href"])
        id_list.append(id[0])
        
    return id_list

# レース詳細情報取得
def scrape_race_info(soup):
    texts = (soup.find("div", attrs={"class": "data_intro"}).find_all("p")[0].text + 
            soup.find("div", attrs={"class": "data_intro"}).find_all("p")[1].text)
    title = soup.find('title').text
    info = re.findall(r'\w+', texts)
    info_dict = {}

    if "G1" in title:
        info_dict["Grade"] = "1"
    elif "G2" in title:
        info_dict["Grade"] = "2"
    elif "G3" in title:
        info_dict["Grade"] = "3"
    else:
        info_dict["Grade"] = "4"

    for text in info:
        if text in ["芝", "ダート"]:
            info_dict["race_type"] = text
        if "障" in text:
            info_dict["race_type"] = "障害"
        if "m" in text:
            info_dict["course_len"] = int(re.findall(r"\d+", text)[0])
        if text in ["良", "稍重", "重", "不良"]:
            info_dict["ground_state"] = text
        if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
            info_dict["weather"] = text
        if "年" in text and "月" in text and "日" in text: 
            info_dict["date"] = text
        # change コース特性追加
        if "右" in text:
            info_dict["course_type"] = "right"
        if "左" in text:
            info_dict["course_type"] = "left"
        if "直線" in text:
            info_dict["course_type"] = "straight"
            
    return info_dict

# レース結果を取得
def scrape_race_results(race_id_list, pre_race_results={}):
    race_results = pre_race_results
    race_infos = {}
    for race_id in race_id_list:
        if race_id in race_results.keys():
            continue
        try:
            url = "https://db.netkeiba.com/race/" + race_id
            df = pd.read_html(url)[0]
            html = requests.get(url)
            html.encoding = "EUC-JP"
            print("scrape_race_results race_id: " + race_id)
            soup = BeautifulSoup(html.text, "html.parser")
            
            if len(df) < 3:
                continue
                
            race_span,flag = scrape_race_span(race_id,len(df))
            if flag != True:
                print("length error",race_id)
                continue
            df["race_span"] = race_span
            
            horse_id_list = scrape_id(soup,"horse")
            jockey_id_list = scrape_id(soup,"jockey")

            if len(horse_id_list) != len(df) or len(jockey_id_list) != len(df):
                continue
                        
            df["horse_id"] = horse_id_list
            df["jockey_id"] = jockey_id_list
            
            # change コースid追加
            df['course_id'] = [int(race_id[4:6])]*len(df)
            
            race_results[race_id] = df

            info_dict = {}
            info_dict = scrape_race_info(soup)
            
            race_infos[race_id] = info_dict
            time.sleep(0.1)
        except IndexError:
            continue
        except Exception as e:
            print(e)
            break
    return race_results,race_infos

# 馬の戦績取得
def scrape_horse_results(horse_id_list, pre_horse_id=[]):
    horse_results = {}
    for horse_id in horse_id_list:
        if horse_id in pre_horse_id:
            continue
        try:
            url = 'https://db.netkeiba.com/horse/' + horse_id
            html = requests.get(url)
            html.encoding = "EUC-JP"
            soup = BeautifulSoup(html.text, "html.parser")
            
            ## add(生産地)
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
        merged_df = pd.concat([self.merge(results, date, n_samples) for date in date_list])
        return merged_df

# 馬の血統取得
def scrape_peds(horse_id_list, pre_peds={}):
    peds = pre_peds
    for horse_id in horse_id_list:
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
    for jockey_id in jockey_id_list:
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

# データ取得手順
def get_race_data(race_id_list,flag):
    print("レース結果取得中")
    results,race_infos = scrape_race_results(race_id_list)
    for key in results:
        results[key].index = [key] * len(results[key])
    results = pd.concat([results[key] for key in results], sort=False)
    df_race_infos = pd.DataFrame(race_infos.values(), index=race_infos.keys())
    results_addinfo = results.merge(df_race_infos,left_index=True,right_index=True,how='inner')
    results_addinfo['date'] = pd.to_datetime(results_addinfo['date'],format='%Y年%m月%d日')
    
    print("馬情報取得中")
    horse_id_list = results_addinfo['horse_id'].unique()
    horse_results = scrape_horse_results(horse_id_list)
    for key in horse_results:
        horse_results[key].index = [key] * len(horse_results[key])
    df_horse_results = pd.concat([horse_results[key] for key in horse_results])
    
    print("ジョッキー情報取得中")
    jockey_id_list = results_addinfo['jockey_id'].unique()
    jockey_results = scrape_jockey_results(jockey_id_list)
    for key in jockey_results:
        jockey_results[key].index = [key] * len(jockey_results[key])
    df_jockey_results = pd.concat([jockey_results[key] for key in jockey_results])
    results_addinfo = results_addinfo.merge(df_jockey_results,left_on='jockey_id',right_index=True,how='left')

    borned_place_list = []
    for i in range(len(results_addinfo)):
        borned_place_list.append(list(set(list(horse_results[results_addinfo['horse_id'][i]]["Borned_place"])))[0])
    results_addinfo["Borned_place"] = borned_place_list
    
    results_addinfo = results_addinfo[~(results_addinfo['着順'].astype(str).str.contains('\D'))]
    drop_lines = list(results_addinfo.query('馬体重 == "計不"').index)
    results_addinfo = results_addinfo.drop(index=drop_lines)

    print("データ結合中")
    hr = HorseResults(df_horse_results)
    results_5R = hr.merge_all(results_addinfo, n_samples=5)
    
    print("血統情報取得中")
    add_blood = add_blood_data(horse_id_list,results_5R)
    if flag == True:
        add_blood.to_pickle('../../../data/raw/race_results_dif_df.pickle')
        return add_blood
    else:
        add_blood.to_pickle('../../../data/raw/race_results_df.pickle')
        return add_blood

# メイン関数
def scraping_netkeiba(parameters):
    print("start scraping_netkeiba")
    # race urlの全量を取得
    get_race_url(parameters)
    
    url_dir = '../../../data/raw/race_url/'

    if not os.path.exists(url_dir):
        os.makedirs(url_dir)
    
    dt_now = datetime.datetime.now()
    race_url_list = []
    for year in range(parameters["scrape_year_start"], parameters["scrape_year_end"]+1):
        if year != dt_now.year:
            for month in range(1,13):
                path = url_dir+str(year)+'-'+str(month)+'.txt'
                with open(path) as f:
                    race_url_list += f.readlines()
        else:
            for month in range(1,dt_now.month):
                path = url_dir+str(year)+'-'+str(month)+'.txt'
                with open(path) as f:
                    race_url_list += f.readlines()
    race_id_list = []
    for url in race_url_list:
        race_id_list.append(url[-14:-2])
        race_id_list = race_id_list[:parameters["scraping_limit"]]
        
    # 初回判定
    if os.path.exists('../../../data/raw/race_results_df.pickle') != True:
        flag = False
        add_blood = get_race_data(race_id_list,flag)
        print("FINISH!!!")
        return

    race_results_df = pd.read_pickle('../../../data/raw/race_results_df.pickle')
    got_race_id_list = set(list(race_results_df.index))
    difference_id_list = set(race_id_list) ^ got_race_id_list
    
    #なぜか失敗する
    if '201305030305' in difference_id_list:
        difference_id_list.remove('201305030305')
    if '201805010107' in difference_id_list:
        difference_id_list.remove('201805010107')
    if '201709050706' in difference_id_list:
        difference_id_list.remove('201709050706')
    if '201808030406' in difference_id_list:
        difference_id_list.remove('201808030406')
    if '201005050810' in difference_id_list:
        difference_id_list.remove('201005050810')
    if '201009040210' in difference_id_list:
        difference_id_list.remove('201009040210')
    if '201006040811' in difference_id_list:
        difference_id_list.remove('201006040811')
    if '201008060611' in difference_id_list:
        difference_id_list.remove('201008060611')
        
    if len(difference_id_list) > 0:
        flag = True
        race_results_dif_df = get_race_data(difference_id_list,flag)
        race_results_df = pd.concat([race_results_df, race_results_dif_df])
        race_results_df.to_pickle('../../../data/raw/race_results_df.pickle')
        return race_results_df
    else:
        flag = False
        return

def main(parameters):
    return scraping_netkeiba(parameters)

if __name__ == "__main__":
    main(parameters)