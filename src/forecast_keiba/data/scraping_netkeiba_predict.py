import pickle
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
import datetime
import requests
from bs4 import BeautifulSoup
import re
import time

class ShutubaTable:
    # 出馬表を作る
    # = race plans
    def __init__(self):
        self.shutuba_table = pd.DataFrame()

    def scrape_shutuba_table(self, race_id_list, predict_date):
        # ベース情報を取ってくる
        options = Options()
        options.binary_location = '/usr/bin/google-chrome'
        options.add_argument('--headless')
        options.add_argument('--window-size=1280,1024')
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
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
            # 詳細情報取得のために、再度対象URLのページを取得する
            html = requests.get(url)
            html.encoding = "EUC-JP"
            soup = BeautifulSoup(html.text, "html.parser")

            # horse_idを取得する
            horse_id_list = []
            horse_soup_list  = soup.find_all("td", attrs={"class": "HorseInfo"})
            for horse_soup in horse_soup_list:
                horse_id_list.append(horse_soup.find("a").get('href')[-10:])

            # jockey_idを取得する
            jockey_id_list = []
            jockey_soup_list  = soup.find_all("td", attrs={"class": "Jockey"})
            for jockey_soup in jockey_soup_list:
                jockey_id_list.append(jockey_soup.find("a").get('href')[-6:-1])

            # horse_idとjockey_idをベースに追加
            # ToDo: race_idごとに作成してmergeするべき（race_id複数対応）
            self.shutuba_table['horse_id'] = horse_id_list
            self.shutuba_table['jockey_id'] = jockey_id_list
            
            # course_len, course_type , weather, race_type, ground_state, dateを取得してベースに追加
            text_race_data = str(soup.find('div',attrs={'class':'RaceData01'}))
            race_data = soup.find('div',attrs={'class':'RaceData01'})

            whether_text = [text_race_data[text_race_data.find("天候")+3:text_race_data.find('<span class="Icon_Weather')]]
            course_type_text = [text_race_data[text_race_data.find("(")+1:text_race_data.find(")")]]
            ground_type_text = [race_data.find_all('span')[0].text]
            ground_state_text = [race_data.find_all('span')[2].text[race_data.find_all('span')[2].text.find(":")+1:]]
            
            race_info = ground_state_text+ ground_type_text + whether_text + course_type_text + predict_date

            info_dict = {}
            race_infos_dict = {}
            for text in race_info:
                if "芝" in text:
                    info_dict["race_type"] = '芝'
                if "ダ" in text:
                    info_dict["race_type"] = 'ダート'
                if "障" in text:
                    info_dict["race_type"] = "障害"
                if "m" in text:
                    info_dict["course_len"] = int(re.findall(r"\d+", text)[0])
                if text in ["良", "稍","稍重", "重", "不良"]:
                    info_dict["ground_state"] = text
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
            race_infos_dict[race_id] = info_dict
            race_infos_df = pd.DataFrame(race_infos_dict.values(), index=race_infos_dict.keys())
            self.shutuba_table = self.shutuba_table.merge(race_infos_df,left_index=True,right_index=True,how='inner')     
            self.shutuba_table['date'] = pd.to_datetime(self.shutuba_table['date'],format='%Y年%m月%d日')

        driver.close()


def scrape_horse_results(horse_id_list, pre_horse_id=[]):
    horse_results = {}
    for horse_id in horse_id_list:
        if horse_id in pre_horse_id:
            continue
        try:
            url = 'https://db.netkeiba.com/horse/' + horse_id
            df = pd.read_html(url)[3]
            if df.columns[0]=='受賞歴':
                df = pd.read_html(url)[4]
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


class HorseResults:
    def __init__(self, horse_results):
        self.horse_results = horse_results[['日付', '着順', '賞金']]
        self.preprocessing()

    def preprocessing(self):
        df = self.horse_results.copy()

        # 着順に数字以外の文字列が含まれているものを取り除く
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        df.dropna(subset=['着順'], inplace=True)
        df['着順'] = df['着順'].astype(int)

        df["date"] = pd.to_datetime(df["日付"])
        df.drop(['日付'], axis=1, inplace=True)

        #賞金のNaNを0で埋める
        df['賞金'].fillna(0, inplace=True)

        self.horse_results = df

    def average(self, horse_id_list, date, n_samples='all'):
        target_df = self.horse_results.loc[horse_id_list]

        #過去何走分取り出すか指定
        if n_samples == 'all':
            filtered_df = target_df[target_df['date'] < date]
        elif n_samples > 0:
            filtered_df = target_df[target_df['date'] < date].\
                sort_values('date', ascending=False).groupby(level=0).head(n_samples)
        else:
            raise Exception('n_samples must be >0')

        average = filtered_df.groupby(level=0)[['着順', '賞金']].mean()
        return average.rename(columns={'着順':'着順_{}R'.format(n_samples), '賞金':'賞金_{}R'.format(n_samples)})
    # change 馬の最高賞金追加
    def max_money(self, horse_id_list, date, n_samples='all'):
        target_df = self.horse_results.loc[horse_id_list]
        
        #過去何走分取り出すか指定
        if n_samples == 'all':
            filtered_df = target_df[target_df['date'] < date]
        elif n_samples > 0:
            filtered_df = target_df[target_df['date'] < date].\
                sort_values('date', ascending=False).groupby(level=0).head(n_samples)
        else:
            raise Exception('n_samples must be >0')
            
        max_money = filtered_df.groupby(level=0)[['賞金']].max()
        return max_money.rename(columns={'賞金':'最高賞金_{}R'.format(n_samples)})

    def merge(self, results, date, n_samples='all'):
        df = results[results['date']==date]
        horse_id_list = df['horse_id']
        merged_df = df.merge(self.average(horse_id_list, date, n_samples), left_on='horse_id',
                             right_index=True, how='left').merge(self.max_money(horse_id_list, date, 'all'), left_on='horse_id',
                             right_index=True, how='left')
        return merged_df

    def merge_all(self, results, n_samples='all'):
        date_list = results['date'].unique()
        merged_df = pd.concat([self.merge(results, date, n_samples) for date in date_list])
        return merged_df


def scraping_netkeiba_predict(parameters):
    print('start scraping_netkeiba_predict')

    predict_race_id = parameters["predict_race_id"]
    predict_date = parameters["predict_date"]
    # データ収集
    st = ShutubaTable()
    st.scrape_shutuba_table([predict_race_id], [predict_date])
    race_plans_df = st.shutuba_table

    # 馬の詳細情報を取得する
    horse_id_list = race_plans_df['horse_id'].unique()

    horse_results_dict = scrape_horse_results(horse_id_list)

    for key in horse_results_dict:
        horse_results_dict[key].index = [key] * len(horse_results_dict[key])
    df_horse_results = pd.concat([horse_results_dict[key] for key in horse_results_dict])

    hr = HorseResults(df_horse_results)

    # 着順_5R, 賞金_5R, 最高賞金_allRを取得してベースに追加
    race_plans_df = hr.merge_all(race_plans_df, n_samples=5)
    race_results_df_predict = race_plans_df

    return race_results_df_predict


def main(parameters):
    return scraping_netkeiba_predict(parameters)

if __name__ == "__main__":
    main(parameters)