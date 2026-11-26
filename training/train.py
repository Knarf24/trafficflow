"""Train and test the arrival forecasts with scikit-learn, then export them to JSON.

Data split, in time order (never shuffled):
  weeks 1-4   history only (features look back up to four weeks)
  weeks 5-10  training
  weeks 11-12 validation: choose between Ridge regression and gradient boosting
  weeks 13-14 test: touched once, after the choice, for the reported scores

The chosen model is refitted on weeks 5-12 and scored on weeks 13-14 next to
the simple baselines. Quantile gradient boosting models give a 10%-90% range.
The data is SYNTHETIC (see training/make_data.py).

Run:  python training/make_data.py && python training/train.py
"""

import json
import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from trafficflow import forecast as fc  # noqa: E402

DATA = os.path.join(ROOT, "data", "synthetic_drive_through.csv")
OUT = os.path.join(ROOT, "web", "data", "models.json")
TRAIN_DAYS = range(28, 70)
VAL_DAYS = range(70, 84)
TEST_DAYS = range(84, 98)
SEED = 7


def matrix(task, days, day_range):
    X, y, where = [], [], []
    for di in day_range:
        for j in range(fc.SLOTS_PER_DAY):
            X.append(fc.feature_row(task, days, di, j))
            y.append(days[di]["arrivals"][j])
            where.append((di, j))
    return np.array(X, dtype=float), np.array(y, dtype=float), where


def gbr(loss="squared_error", alpha=0.5):
    kw = {"alpha": alpha} if loss == "quantile" else {}
    return GradientBoostingRegressor(loss=loss, n_estimators=250, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=20, random_state=SEED, **kw)


def forest():
    return RandomForestRegressor(n_estimators=60, max_depth=9, min_samples_leaf=10, random_state=SEED, n_jobs=-1)


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def export_gbr(model):
    trees = []
    for est in model.estimators_[:, 0]:
        t = est.tree_
        trees.append({
            "f": [int(v) for v in t.feature],
            "t": [round(float(v), 6) for v in t.threshold],
            "l": [int(v) for v in t.children_left],
            "r": [int(v) for v in t.children_right],
            "v": [round(float(v), 6) for v in t.value[:, 0, 0]],
        })
    init = float(np.asarray(model.init_.constant_).ravel()[0])
    return {"kind": "gbr", "init": init, "lr": float(model.learning_rate), "trees": trees}


def export_forest(model):
    trees = []
    for est in model.estimators_:
        t = est.tree_
        trees.append({
            "f": [int(v) for v in t.feature],
            "t": [round(float(v), 5) for v in t.threshold],
            "l": [int(v) for v in t.children_left],
            "r": [int(v) for v in t.children_right],
            "v": [round(float(v), 4) for v in t.value[:, 0, 0]],
        })
    return {"kind": "forest", "trees": trees}


def export_ridge(model):
    scaler, reg = model.steps[0][1], model.steps[1][1]
    return {"kind": "ridge", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": reg.coef_.tolist(), "intercept": float(reg.intercept_)}


def scores(y, p):
    e = p - y
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e))}


def by_period(where, days, y, preds):
    """Error in rush hours vs quiet hours, and on special days."""
    groups = {"rush hours (11:00-14:00, 17:00-19:30)": [], "quiet hours": [], "rainy slots": [],
              "promotion days": [], "weekends": []}
    for k, (di, j) in enumerate(where):
        m = days[di]["minute"][j]
        rush = 660 <= m < 840 or 1020 <= m < 1170
        groups["rush hours (11:00-14:00, 17:00-19:30)" if rush else "quiet hours"].append(k)
        if days[di]["rain_mm"][j] > 0:
            groups["rainy slots"].append(k)
        if days[di]["promo_day"]:
            groups["promotion days"].append(k)
        if days[di]["weekday"] >= 5:
            groups["weekends"].append(k)
    out = {}
    for name, idx in groups.items():
        if idx:
            out[name] = {"slots": len(idx), **{m: scores(y[idx], p[idx])["mae"] for m, p in [(m, v) for m, v in preds.items()]}}
    return out


def run_task(task, days):
    Xtr, ytr, _ = matrix(task, days, TRAIN_DAYS)
    Xva, yva, _ = matrix(task, days, VAL_DAYS)
    candidates = {"ridge": ridge(), "random_forest": forest(), "gradient_boosting": gbr()}
    val = {}
    for name, m in candidates.items():
        m.fit(Xtr, ytr)
        val[name] = scores(yva, np.clip(m.predict(Xva), 0, None))
    chosen = min(val, key=lambda k: val[k]["mae"])

    # refit on training + validation weeks, then score the untouched test weeks
    fit_days = range(TRAIN_DAYS.start, VAL_DAYS.stop)
    Xfit, yfit, _ = matrix(task, days, fit_days)
    Xte, yte, where = matrix(task, days, TEST_DAYS)
    final = {"ridge": ridge(), "random_forest": forest(), "gradient_boosting": gbr()}
    preds = {}
    for name, m in final.items():
        m.fit(Xfit, yfit)
        preds[name] = np.clip(m.predict(Xte), 0, None)
    for b in fc.BASELINES[task]:
        preds[b] = np.array([fc.baseline(task, b, days, di, j) for di, j in where])
    low = gbr("quantile", 0.1).fit(Xfit, yfit)
    high = gbr("quantile", 0.9).fit(Xfit, yfit)
    lo, hi = low.predict(Xte), high.predict(Xte)
    coverage = float(np.mean((yte >= np.minimum(lo, preds[chosen])) & (yte <= np.maximum(hi, preds[chosen]))))

    exporters = {"gradient_boosting": export_gbr, "random_forest": export_forest, "ridge": export_ridge}
    point = exporters[chosen](final[chosen])
    # which inputs matter: how much worse the chosen model gets on the test weeks
    # when one input is shuffled (permutation importance)
    imp = permutation_importance(final[chosen], Xte, yte, scoring="neg_mean_absolute_error",
                                 n_repeats=5, random_state=SEED)
    importance = sorted(({"feature": f, "mae_increase": float(m), "sd": float(sd)}
                         for f, m, sd in zip(fc.FEATURES[task], imp.importances_mean, imp.importances_std)),
                        key=lambda r: -r["mae_increase"])
    test = {name: scores(yte, p) for name, p in preds.items()}
    best_baseline = min(fc.BASELINES[task], key=lambda b: test[b]["mae"])
    print(f"\n[{task}] validation MAE:", {k: round(v["mae"], 3) for k, v in val.items()}, "-> chose", chosen)
    for name, s in sorted(test.items(), key=lambda kv: kv[1]["mae"]):
        print(f"  test {name:24s} MAE {s['mae']:.3f}  RMSE {s['rmse']:.3f}  bias {s['bias']:+.3f}")
    print(f"  10-90% range held the actual count {coverage:.1%} of the time")
    return {
        "features": fc.FEATURES[task],
        "chosen": chosen,
        "point": point,
        "low": export_gbr(low),
        "high": export_gbr(high),
        "validation": val,
        "test": test,
        "best_baseline": best_baseline,
        "improvement_vs_best_baseline": 1 - test[chosen]["mae"] / test[best_baseline]["mae"],
        "interval_coverage": coverage,
        "importance": importance,
        "by_period": by_period(where, days, yte, {k: preds[k] for k in list(final) + [best_baseline]}),
        "test_mean_arrivals": float(np.mean(yte)),
    }


def main():
    with open(DATA) as f:
        days = fc.load_days(f.read())
    bundle = {
        "data": "synthetic",
        "split": {"history": [days[0]["date"], days[27]["date"]],
                  "train": [days[TRAIN_DAYS.start]["date"], days[TRAIN_DAYS.stop - 1]["date"]],
                  "validation": [days[VAL_DAYS.start]["date"], days[VAL_DAYS.stop - 1]["date"]],
                  "test": [days[TEST_DAYS.start]["date"], days[TEST_DAYS.stop - 1]["date"]]},
        "test_days": [TEST_DAYS.start, TEST_DAYS.stop],
        "baseline_names": fc.BASELINES,
        "tasks": {t: run_task(t, days) for t in ("next15", "day_ahead")},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(bundle, f, separators=(",", ":"))
    print(f"\nwrote {os.path.normpath(OUT)} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
