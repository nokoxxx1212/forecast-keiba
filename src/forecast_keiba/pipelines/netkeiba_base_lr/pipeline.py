from kedro.pipeline import Pipeline, node
import sys
import os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR + "/../../")
from data import scraping_netkeiba
from features import preprocess_race_results_base
from models import train_lr
from models import valid_lr
from models import predict_lr

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=scraping_netkeiba.scraping_netkeiba,
                inputs=["parameters"],
                outputs="race_results_df"
            ),
            node(
                func=preprocess_race_results_base.preprocess_race_results_base,
                inputs=["race_results_df", "parameters"],
                outputs="race_results_df_processed_base"
            ),
            node(
                func=train_lr.train_lr,
                inputs=["race_results_df_processed_base", "parameters"],
                outputs="model_lr"
            ),
            node(
                func=valid_lr.valid_lr,
                inputs=["race_results_df_processed_valid", "model_lr", "parameters"],
                outputs=None
            ),
            node(
                func=predict_lr.predict_lr,
                inputs=["model_lr", "parameters"],
                outputs=None
            ),
        ]
    )
