import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import lightgbm as lgb
from sklearn.metrics import ndcg_score
from typing import Mapping, Optional, Sequence


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.lightgbm.autolog(log_models=False)


def train_lightgbm(
    train_data: lgb.Dataset,
    val_data: lgb.Dataset,
    params: Mapping,
    num_boost_round: int,
) -> lgb.Booster:
    return lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "validation"],
        num_boost_round=num_boost_round,
    )


def compute_group_ndcg(
    scored_df,
    score_column: str = "pred",
    relevance_column: str = "relevance",
    group_columns: Sequence[str] = None,
) -> tuple[float, int]:
    group_columns = group_columns or ["location_id", "heat_num", "date_id"]
    ndcg_scores = []

    for _, group in scored_df.groupby(group_columns):
        if len(group) < 2:
            continue

        true_relevance = group[relevance_column].to_numpy().reshape(1, -1)
        predicted_scores = group[score_column].to_numpy().reshape(1, -1)

        ndcg_scores.append(ndcg_score(true_relevance, predicted_scores))

    if not ndcg_scores:
        raise ValueError("No validation heats had enough horses to calculate NDCG.")

    return float(np.mean(ndcg_scores)), len(ndcg_scores)


def plot_feature_importance(model: lgb.Booster) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    lgb.plot_importance(
        model,
        importance_type="gain",
        ax=ax,
        title="LightGBM Feature Importance (Gain)",
    )
    fig.tight_layout()
    return fig


def save_model(model: lgb.Booster, output_path: str) -> str:
    model.save_model(output_path)
    return output_path
