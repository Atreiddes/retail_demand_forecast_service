"""Структурный иерархический байес (numpyro NUTS) под M5 weekly demand.

Прогноз = level[ряд] * exp(b_c[g]*якорь(h) + тренд[g]*демпф + сезонность[g]): объём
несёт MA-уровень ряда, форму учит байес на уровне группы. Без утечек: level, ratio и
сезонность считаются только по train. Точечный прогноз. Требует numpyro + jax.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Глушим только шум numpyro/jax/arviz, не скрывая pandas SettingWithCopy/RuntimeWarning.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

TCOL = "week_start_date"
SERIES = "id"
VALUE = "units"

K_FOURIER = 5          # гармоник сезонности
PHI_DAMP = 0.6         # демпфирование экстраполяции тренда за train
ANCHOR_PHI = 0.85      # якорь b_c по горизонту
SB_SCALE = 0.15        # стягивание b_c (HalfNormal scale): сильнее к global, регуляризация групп
SEAS_PRIOR = 1.0       # амплитуда сезонности
LEVEL_K = 8            # окно MA-уровня ряда
NUTS_WARMUP = 400
NUTS_SAMPLES = 600


def _woy(s):
    return s.dt.isocalendar().week.astype(int).clip(1, 53).to_numpy()


def _week_ord(s):
    return (s.astype("int64") // (7 * 24 * 3600 * 10**9)).to_numpy()


def _fourier(woy, K=K_FOURIER):
    woy = np.asarray(woy, float)
    cols = []
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * woy / 52.0))
        cols.append(np.cos(2 * np.pi * k * woy / 52.0))
    return np.stack(cols, axis=1)  # (N, 2K)


def series_level(train, k=LEVEL_K):
    """MA-k уровень на ряд: среднее последних k train-недель. Без утечек."""
    w = sorted(train[TCOL].unique())
    return (
        train[train[TCOL].isin(w[-k:])]
        .groupby(SERIES, observed=True)[VALUE]
        .mean()
        .rename("lvl")
        .reset_index()
    )


def _group_ratio(train, level, group):
    """ratio[g, t] = sum(units) / sum(level) по группе и неделе (для фита формы). Без утечек."""
    tr = train.merge(level, on=SERIES, how="left")
    tr["lvl"] = tr["lvl"].fillna(0.0)
    g = (
        tr.groupby([group, TCOL], observed=True)
        .agg(y=(VALUE, "sum"), base=("lvl", "sum"))
        .reset_index()
    )
    g = g[g["base"] > 0].copy()
    g["ratio"] = (g["y"] / g["base"]).clip(0.05, 20.0)
    return g


def _shape_model(gid, tnorm, four, logratio, n_group):
    """Иерархия global -> group мультипликативной формы. logratio = log(sum y / sum level)."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    Kf = four.shape[1]
    b_g = numpyro.sample("b_g", dist.Normal(0.0, 0.3))
    s_b = numpyro.sample("s_b", dist.HalfNormal(SB_SCALE))
    b_c = numpyro.sample("b_c", dist.Normal(b_g, s_b).expand([n_group]).to_event(1))
    t_g = numpyro.sample("t_g", dist.Normal(0.0, 0.15))
    s_t = numpyro.sample("s_t", dist.HalfNormal(0.15))
    trend_c = numpyro.sample("trend_c", dist.Normal(t_g, s_t).expand([n_group]).to_event(1))
    s_seas = numpyro.sample("s_seas", dist.HalfNormal(SEAS_PRIOR))
    seas = numpyro.sample("seas", dist.Normal(0.0, s_seas).expand([n_group, Kf]).to_event(2))
    sig = numpyro.sample("sig", dist.HalfNormal(0.5))
    mu = b_c[gid] + trend_c[gid] * tnorm + jnp.sum(four * seas[gid], axis=1)
    numpyro.sample("lik", dist.Normal(mu, sig).to_event(1), obs=logratio)


def fit_shape(train, level, group_map, group):
    """Фит байесовой формы на групповом ratio. Возвращает posterior-сэмплы."""
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    g = _group_ratio(train, level, group)
    tr_end = train[TCOL].max()
    g["tnorm"] = (_week_ord(g[TCOL]) - _week_ord(pd.Series([tr_end]))[0]) / 52.0
    g["gid"] = g[group].map(group_map).astype(int)
    four = _fourier(_woy(g[TCOL]))
    nuts = NUTS(_shape_model, target_accept_prob=0.9)
    mcmc = MCMC(nuts, num_warmup=NUTS_WARMUP, num_samples=NUTS_SAMPLES,
                num_chains=1, progress_bar=False)
    mcmc.run(
        jax.random.PRNGKey(0),
        jnp.array(g["gid"].to_numpy()),
        jnp.array(g["tnorm"].to_numpy(dtype=float)),
        jnp.array(four),
        jnp.array(np.log(g["ratio"].to_numpy())),
        len(group_map),
    )
    return {k: np.array(v) for k, v in mcmc.get_samples().items()}


def predict(post, test, level, group_map, group, tr_end, horizon=4):
    """Точечный прогноз: level[ряд] * exp(b_c*якорь + тренд*демпф + сезонность).

    Якорь нормирован на horizon: b_c нарастает от ~0 у origin до полного к концу горизонта.
    """
    te = test.merge(level, on=SERIES, how="left")
    te["lvl"] = te["lvl"].fillna(0.0)
    gid = te[group].map(group_map).astype(int).to_numpy()
    tnorm = (_week_ord(te[TCOL]) - _week_ord(pd.Series([tr_end]))[0]) / 52.0
    h = tnorm * 52.0
    bramp = (1.0 - ANCHOR_PHI**h) / (1.0 - ANCHOR_PHI**horizon)

    b = np.median(post["b_c"], axis=0)[gid] * bramp
    tr = np.median(post["trend_c"], axis=0)[gid] * (PHI_DAMP * tnorm)
    four = _fourier(_woy(te[TCOL]))
    seas_med = np.median(post["seas"], axis=0)            # (G, 2K)
    seas = np.einsum("nk,nk->n", four, seas_med[gid])     # (N,)

    factor = np.exp(b + tr + seas)
    pred = np.clip(te["lvl"].to_numpy() * factor, 0, None)
    out = te[[SERIES, TCOL]].copy()
    out["pred"] = pred
    return out


def forecast(train, test, group="dept_id", horizon=4):
    """Полный цикл: fit на train -> точечный прогноз на test (ряды x недели)."""
    group_map = {c: i for i, c in enumerate(sorted(train[group].astype(str).unique()))}
    train = train.copy()
    test = test.copy()
    train[group] = train[group].astype(str)
    test[group] = test[group].astype(str)
    level = series_level(train)
    post = fit_shape(train, level, group_map, group)
    tr_end = train[TCOL].max()
    return predict(post, test, level, group_map, group, tr_end, horizon=horizon)
