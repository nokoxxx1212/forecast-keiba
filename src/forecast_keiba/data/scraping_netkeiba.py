import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
import re

def exec_scraping_netkeiba_race_results(race_id_list):
    """
    netkeiba.comのレースIDのリストを渡して、それらをまとめて{'レースID', 結果のDataFrame}という形式の辞書型に格納する
    ex. race_results['201901010101']
        -> df 着順 枠番 馬番 馬名 性齢 斤量 騎手 タイム 着差 単勝 人気 馬体重 調教師 horse_id jockey_id course_id
    """
    race_results_dict = {}
    for race_id in race_id_list:
        try:
            # ベース情報を取ってくる
            url = 'https://db.netkeiba.com/race/' + race_id
            race_results_dict[race_id] = pd.read_html(url)[0]

            # 詳細情報取得のために、再度対象URLのページを取得する
            html = requests.get(url)
            html.encoding = "EUC-JP"
            soup = BeautifulSoup(html.text, "html.parser")

            # horse_idを取得する
            horse_id_list = []
            horse_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all("a", attrs={"href": re.compile("^/horse")})
            for a in horse_a_list:
                horse_id = re.findall(r"\d+", a["href"])
                horse_id_list.append(horse_id[0])

            # jockey_idを取得する
            jockey_id_list = []
            jockey_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                "a", attrs={"href": re.compile("^/jockey")}
            )
            for a in jockey_a_list:
                jockey_id = re.findall(r"\d+", a["href"])
                jockey_id_list.append(jockey_id[0])

            # horse_idとjockey_idをベースに追加
            race_results_dict[race_id]["horse_id"] = horse_id_list
            race_results_dict[race_id]["jockey_id"] = jockey_id_list

            # course_idをベースに追加
            race_results_dict[race_id]['course_id'] = [int(race_id[4:6])]*len(horse_id_list)

            # course_len, course_type , weather, race_type, ground_state, dateを取得してベースに追加
            texts = (
                soup.find("div", attrs={"class": "data_intro"}).find_all("p")[0].text
                + soup.find("div", attrs={"class": "data_intro"}).find_all("p")[1].text
            )
            info = re.findall(r'\w+', texts)
            for text in info:
                if text in ["芝", "ダート"]:
                    race_results_dict[race_id]["race_type"] = text
                if "障" in text:
                    race_results_dict[race_id]["race_type"] = "障害"
                if "m" in text:
                    race_results_dict[race_id]["course_len"] = int(re.findall(r"\d+", text)[0])
                if text in ["良", "稍重", "重", "不良"]:
                    race_results_dict[race_id]["ground_state"] = text
                if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                    race_results_dict[race_id]["weather"] = text
                if "年" in text:
                    race_results_dict[race_id]["date"] = text
                    race_results_dict[race_id]['date'] = pd.to_datetime(race_results_dict[race_id]['date'],format='%Y年%m月%d日')
                # change コース特性追加
                if "右" in text:
                    race_results_dict[race_id]["course_type"] = "right"
                if "左" in text:
                    race_results_dict[race_id]["course_type"] = "left"
                if "直線" in text:
                    race_results_dict[race_id]["course_type"] = "straight"

            time.sleep(0.1)

        except IndexError:
            continue
        except:
            break
    return race_results_dict


def scrape_horse_results(horse_id_list):
    horse_results = {}
    for horse_id in horse_id_list:
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

        # 賞金のNaNを0で埋める
        df['賞金'].fillna(0, inplace=True)

        self.horse_results = df

    def average(self, horse_id_list, date, n_samples='all'):
        target_df = self.horse_results.loc[horse_id_list]

        # 過去何走分取り出すか指定
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

        # 過去何走分取り出すか指定
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


def scraping_netkeiba(parameters):
    race_id_list = []

    for place in range(1,11):
        for kai in range(1,6):
            for day in range(1,9):
                for r in range(1,13):
                    race_id = '2019' + str(place).zfill(2) + str(kai).zfill(2) + str(day).zfill(2) + str(r).zfill(2)
                    race_id_list.append(race_id)
    # -> ['201901010101', '201901010102', ,,, ]

    # スクレイピング実行
    race_results_dict = exec_scraping_netkeiba_race_results(race_id_list[0:parameters["scraping_limit"]])

    # 表として見やすいように、dfのindexにrace idを入れる
    for key in race_results_dict.keys():
        race_results_dict[key].index = [key]*len(race_results_dict[key])

    # 各レース結果のdfを1つに結合する
    race_results_df = pd.concat((race_results_dict[key] for key in race_results_dict.keys()),sort=False)

    # 馬の詳細情報を取得する
    horse_id_list = race_results_df['horse_id'].unique()

    horse_results_dict = scrape_horse_results(horse_id_list)

    for key in horse_results_dict:
        horse_results_dict[key].index = [key] * len(horse_results_dict[key])
    df_horse_results = pd.concat([horse_results_dict[key] for key in horse_results_dict])

    hr = HorseResults(df_horse_results)

    # 着順_5R, 賞金_5R, 最高賞金_allRを取得してベースに追加
    race_results_df = hr.merge_all(race_results_df, n_samples=5)

    return race_results_df


def main(parameters):
    return scraping_netkeiba(parameters)

if __name__ == "__main__":
    main(parameters)