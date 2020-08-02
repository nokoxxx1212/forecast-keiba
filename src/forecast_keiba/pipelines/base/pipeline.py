from kedro.pipeline import Pipeline, node
import sys
import os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR + "/../../")
from data import scraping_netkeiba
from data import scraping_netkeiba_predict
from features import preprocess_race_results
from features import preprocess_race_results_predict
from models import train_lr
from models import train_rf
from models import train_lightgbm
from models import valid_lr
from models import predict_lr
from models import predict_rf
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
                func=train_lr.train_lr,
                inputs=["race_results_df_processed", "parameters"],
                outputs="model_lr"
            ),
            node(
                func=train_rf.train_rf,
                inputs=["race_results_df_processed", "parameters"],
                outputs="model_rf"
            ),
            node(
                func=train_lightgbm.train_lightgbm,
                inputs=["race_results_df_processed", "parameters"],
                outputs="model_lightgbm"
            ),
#            node(
#                func=valid_lr.valid_lr,
#                inputs=["race_results_df_processed_valid", "model_lr", "parameters"],
#                outputs=None
#            ),
            node(
                func=scraping_netkeiba_predict.scraping_netkeiba_predict,
                inputs=["parameters"],
                outputs="race_results_df_predict"
            ),
            node(
                func=preprocess_race_results_predict.preprocess_race_results_predict,
                inputs=["race_results_df_predict", "race_results_df_processed", "parameters"],
                outputs="race_results_df_processed_predict"
            ),
            node(
                func=predict_lr.predict_lr,
                inputs=["model_lr", "race_results_df_processed_predict", "parameters"],
                outputs=None
            ),
            node(
                func=predict_rf.predict_rf,
                inputs=["model_rf", "race_results_df_processed_predict", "parameters"],
                outputs=None
            ),
            node(
                func=predict_lightgbm.predict_lightgbm,
                inputs=["model_lightgbm", "race_results_df_processed_predict", "parameters"],
                outputs=None
            ),
        ]
    )
