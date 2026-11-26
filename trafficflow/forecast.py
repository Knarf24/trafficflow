"""Arrival forecasting: features, baselines and model inference.

Two forecasting tasks, both predicting the number of cars arriving in a
15-minute slot:

* "next15": at the end of a slot, predict the next slot. Uses what has just
  happened (the last hour of arrivals) plus calendar and weather.
* "day_ahead": the evening before, predict every slot of tomorrow. Uses only
  calendar, weather and the same slots in earlier weeks.

The models are trained with scikit-learn in training/train.py and exported to
JSON. This module rebuilds the exact same features and runs the exported
models in plain Python, so forecasting works in the browser without
scikit-learn. Both tasks have simple baselines the models must beat.
"""

from __future__ import annotations

import csv
import io
import math

SLOTS_PER_DAY = 68            # 06:00 to 23:00 in 15-minute slots
HISTORY_DAYS = 28             # four weeks needed before a day can be forecast

BASE_FEATURES = ["minute_of_day", "weekday", "weekend", "holiday", "promo_day", "school_day",
                 "temp_c", "rain_mm", "tod_s1", "tod_c1", "tod_s2", "tod_c2", "tod_s3", "tod_c3"]
FEATURES = {
    "next15": BASE_FEATURES + ["lag1", "lag2", "lag3", "lag4", "yesterday", "last_week",
                               "last_week_next", "today_vs_last_week"],
    "day_ahead": BASE_FEATURES + ["last_week", "two_weeks", "mean4", "last_week_smooth"],
}
BASELINES = {
    "next15": {
        "same_as_last_15_min": "Same as the last 15 minutes",
        "same_time_last_week": "Same time last week",
    },
    "day_ahead": {
        "same_time_last_week": "Same time last week",
        "average_last_4_weeks": "Average of the same time over the last 4 weeks",
    },
}


# ---------------------------------------------------------------- data

def load_days(csv_text):
    """CSV text -> list of days, each with per-slot lists."""
    days = []
    by_date = {}
    for row in csv.DictReader(io.StringIO(csv_text)):
        d = by_date.get(row["date"])
        if d is None:
            d = {"date": row["date"], "weekday": int(row["weekday"]), "holiday": int(row["holiday"]),
                 "promo_day": int(row["promo_day"]), "school_day": int(row["school_day"]),
                 "minute": [], "arrivals": [], "temp_c": [], "rain_mm": []}
            by_date[row["date"]] = d
            days.append(d)
        d["minute"].append(int(row["minute_of_day"]))
        d["arrivals"].append(int(row["arrivals"]))
        d["temp_c"].append(float(row["temp_c"]))
        d["rain_mm"].append(float(row["rain_mm"]))
    for d in days:
        if len(d["arrivals"]) != SLOTS_PER_DAY:
            raise ValueError(f"{d['date']} has {len(d['arrivals'])} slots, expected {SLOTS_PER_DAY}")
    return days


def _base(day, j):
    m = day["minute"][j] + 7.5
    x = 2 * math.pi * m / 1440.0
    wd = day["weekday"]
    return [day["minute"][j], wd, 1 if wd >= 5 else 0, day["holiday"], day["promo_day"], day["school_day"],
            day["temp_c"][j], day["rain_mm"][j],
            math.sin(x), math.cos(x), math.sin(2 * x), math.cos(2 * x), math.sin(3 * x), math.cos(3 * x)]


def _series_before(days, di, j, k):
    """The k-th slot before slot j of day di (walking back across nights)."""
    pos = di * SLOTS_PER_DAY + j - k
    return days[pos // SLOTS_PER_DAY]["arrivals"][pos % SLOTS_PER_DAY]


def features_next15(days, di, j):
    d = days[di]
    lw = days[di - 7]["arrivals"]
    so_far = sum(d["arrivals"][:j])
    lw_so_far = sum(lw[:j])
    ratio = (so_far + 5.0) / (lw_so_far + 5.0)
    return _base(d, j) + [
        _series_before(days, di, j, 1), _series_before(days, di, j, 2),
        _series_before(days, di, j, 3), _series_before(days, di, j, 4),
        days[di - 1]["arrivals"][j], lw[j], lw[min(j + 1, SLOTS_PER_DAY - 1)], ratio,
    ]


def features_day_ahead(days, di, j, day=None):
    """`day` lets a made-up future day (calendar + weather) be forecast."""
    d = day or days[di]
    w = [days[di - 7 * k]["arrivals"] for k in (1, 2, 3, 4)]
    lw = w[0]
    smooth = sum(lw[max(0, j - 1):j + 2]) / len(lw[max(0, j - 1):j + 2])
    return _base(d, j) + [lw[j], w[1][j], sum(x[j] for x in w) / 4.0, smooth]


def baseline(task, name, days, di, j):
    if name == "same_as_last_15_min":
        return float(_series_before(days, di, j, 1))
    if name == "same_time_last_week":
        return float(days[di - 7]["arrivals"][j])
    if name == "average_last_4_weeks":
        return sum(days[di - 7 * k]["arrivals"][j] for k in (1, 2, 3, 4)) / 4.0
    raise ValueError(name)


def feature_row(task, days, di, j, day=None):
    if task == "next15":
        return features_next15(days, di, j)
    return features_day_ahead(days, di, j, day)


# ---------------------------------------------------------------- exported models

def predict_one(model, x):
    if model["kind"] == "gbr":
        total = model["init"]
        lr = model["lr"]
        for tree in model["trees"]:
            node = 0
            left, right, feat, thr, val = tree["l"], tree["r"], tree["f"], tree["t"], tree["v"]
            while left[node] != -1:
                node = left[node] if x[feat[node]] <= thr[node] else right[node]
            total += lr * val[node]
        return total
    if model["kind"] == "forest":
        total = 0.0
        for tree in model["trees"]:
            node = 0
            left, right, feat, thr, val = tree["l"], tree["r"], tree["f"], tree["t"], tree["v"]
            while left[node] != -1:
                node = left[node] if x[feat[node]] <= thr[node] else right[node]
            total += val[node]
        return total / len(model["trees"])
    if model["kind"] == "ridge":
        return model["intercept"] + sum(
            c * (v - m) / s for c, v, m, s in zip(model["coef"], x, model["mean"], model["scale"]))
    raise ValueError(model["kind"])


def forecast_day(bundle, task, days, di, day=None):
    """Forecast (point, low, high) for every slot of day `di` with the exported models."""
    models = bundle["tasks"][task]
    point, lo, hi = [], [], []
    for j in range(SLOTS_PER_DAY):
        x = feature_row(task, days, di, j, day)
        p = max(0.0, predict_one(models["point"], x))
        a = max(0.0, predict_one(models["low"], x))
        b = max(0.0, predict_one(models["high"], x))
        point.append(p)
        lo.append(min(a, p))
        hi.append(max(b, p))
    return point, lo, hi


def errors(actual, predicted):
    n = len(actual)
    e = [p - a for a, p in zip(actual, predicted)]
    mae = sum(abs(v) for v in e) / n
    rmse = math.sqrt(sum(v * v for v in e) / n)
    return {"mae": mae, "rmse": rmse, "bias": sum(e) / n}
