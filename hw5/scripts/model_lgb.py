"""LightGBM (Tweedie) под M5 weekly + тюнинг Optuna по валидационному WRMSSE.

Одна модель на всех рядах, признаки из features.py. Tweedie-loss под прерывистый
спрос, sample_weight по revenue для выравнивания под WRMSSE (важные ряды весомее).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features_direct import CAT_FEATURES, FEATURES, TARGET, TCOL

DEFAULT = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "num_threads": 0,
    "verbose": -1,
    "force_col_wise": True,
}


def _weight(df):
    return np.log1p(df["revenue"].clip(lower=0).fillna(0).to_numpy())


def _dataset(df, feats=None):
    import lightgbm as lgb

    return lgb.Dataset(
        df[feats or FEATURES], label=df[TARGET], weight=_weight(df),
        categorical_feature=CAT_FEATURES, free_raw_data=False,
    )


def train(train_df, params=None, num_boost_round=600, valid_df=None, early=50, feats=None):
    import lightgbm as lgb

    p = {**DEFAULT, **(params or {})}
    if p.get("objective") != "tweedie":
        p.pop("tweedie_variance_power", None)
    dtrain = _dataset(train_df, feats)
    callbacks = [lgb.log_evaluation(0)]
    valid_sets = [dtrain]
    if valid_df is not None:
        valid_sets.append(_dataset(valid_df, feats))
        callbacks.append(lgb.early_stopping(early, verbose=False))
    return lgb.train(p, dtrain, num_boost_round=num_boost_round,
                     valid_sets=valid_sets, callbacks=callbacks)


def predict(model, df, feats=None):
    pred = np.clip(model.predict(df[feats or FEATURES], num_iteration=model.best_iteration), 0, None)
    out = df[["id", TCOL]].copy()
    out["pred"] = pred
    return out


def tune(train_df, valid_df, scorer, valid_actual, n_trials=25, seed=42, metric="WRMSSE"):
    """Optuna TPE: минимизируем валидационную метрику (по умолчанию WRMSSE). Возвращает (best_params, study)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.05, 1.6),
            "num_leaves": trial.suggest_int("num_leaves", 31, 383),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 1000),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        }
        model = train(train_df, params, num_boost_round=2000, valid_df=valid_df, early=50)
        preds = predict(model, valid_df)
        return scorer.score(valid_actual, preds)[metric]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**DEFAULT, **study.best_params}, study
