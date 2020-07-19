import pandas as pd
import time

def exec_scraping_netkeiba_race_results(race_id_list: list) -> dict:
    """
    netkeiba.comのレースIDのリストを渡して、それらをまとめて{'レースID', 結果のDataFrame}という形式の辞書型に格納する
    ex. race_results['201901010101']
        -> df 着順 枠番 馬番 馬名 性齢 斤量 騎手 タイム 着差 単勝 人気 馬体重 調教師
    """
    race_results_dict = {}
    for race_id in race_id_list:
        try:
            url = 'https://db.netkeiba.com/race/' + race_id
            race_results_dict[race_id] = pd.read_html(url)[0]
            time.sleep(0.1)
        except IndexError:
            continue
        except:
            break
    return race_results_dict


def scraping_netkeiba():
    # race id を生成する（規則的に生成できる）
    # todo: 2019年に絞っている
    race_id_list = []

    for place in range(1,11):
        for kai in range(1,6):
            for day in range(1,9):
                for r in range(1,13):
                    race_id = '2019' + str(place).zfill(2) + str(kai).zfill(2) + str(day).zfill(2) + str(r).zfill(2)
                    race_id_list.append(race_id)
    # -> ['201901010101', '201901010102', ,,, ]

    # スクレイピング実行
    # todo: listを絞っている
    race_results_dict = exec_scraping_netkeiba_race_results(race_id_list[0:50])

    # 表として見やすいように、dfのindexにrace idを入れる
    for key in race_results_dict.keys():
        race_results_dict[key].index = [key]*len(race_results_dict[key])

    # 各レース結果のdfを1つに結合する
    race_results_df = pd.concat((race_results_dict[key] for key in race_results_dict.keys()),sort=False)

    return race_results_df


def main():
    return scraping_netkeiba()

if __name__ == "__main__":
    main()