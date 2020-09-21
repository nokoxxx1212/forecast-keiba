from kedro.pipeline import Pipeline, node
import sys
import os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR + "/../../")
from data import scraping_netkeiba
from features import preprocess_race_results
from models import train_lightgbm
from models import predict_lightgbm

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=scraping_netkeiba.scraping_netkeiba,
                inputs=["parameters"],
                outputs="race_results_df"
            ),
            node(
                func=preprocess_race_results.preprocess_race_results,
                inputs=["race_results_df", "parameters"],
                outputs="race_results_df_processed"
            ),
            node(
                func=train_lightgbm.train_lightgbm,
                inputs=["race_results_df_processed", "parameters"],
                outputs="model_lightgbm"
            ),
            node(
                func=predict_lightgbm.predict_lightgbm,
                inputs=["model_lightgbm", "parameters"],
                outputs=None
            ),
        ]
    )
