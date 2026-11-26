"""One entry point for the web page: call(name, json_text) -> json_text.

The page runs this package in the browser with Pyodide (in a Web Worker), so all
simulation, forecasting and planning is done by the same Python code that the
tests check.
"""

from __future__ import annotations

import json

from . import forecast as fc
from . import planner, sim

STATE = {"days": None, "bundle": None}


def _load(args):
    STATE["days"] = fc.load_days(args["csv"])
    STATE["bundle"] = json.loads(args["models"])
    days = STATE["days"]
    return {"n_days": len(days), "first": days[0]["date"], "last": days[-1]["date"],
            "test_days": STATE["bundle"]["test_days"],
            "dates": [[d["date"], d["weekday"], d["holiday"], d["promo_day"]] for d in days]}


def _compare(args):
    return sim.compare(args["current"], args["proposed"], args["rates"], reps=int(args.get("reps", 30)),
                       seed=int(args.get("seed", 1)), slot_min=float(args.get("slot_min", 15)),
                       initial_queue=int(args.get("initial_queue", 0)))


def _replay(args):
    """One traced run, for the animated lane view."""
    rates = args["rates"]
    horizon = float(args.get("slot_min", 15)) * len(rates)
    cars = sim.fixed_like(int(args.get("initial_queue", 0)), int(args.get("seed", 1))) + \
        sim.make_cars(rates, int(args.get("seed", 1)), float(args.get("slot_min", 15)))
    res = sim.simulate(args["params"], cars, horizon, trace=True)
    keep = ["arrive", "into", "start", "done", "leave", "balked"]
    res["cars"] = [{k: c[k] for k in keep} for c in res["cars"]]
    res.pop("curve", None)
    return res


def _day(args):
    """Actual counts, both forecasts and the baselines for one day of the dataset."""
    days, bundle = STATE["days"], STATE["bundle"]
    di = int(args["day"])
    out = {"date": days[di]["date"], "weekday": days[di]["weekday"], "actual": days[di]["arrivals"],
           "minute": days[di]["minute"], "rain_mm": days[di]["rain_mm"], "temp_c": days[di]["temp_c"],
           "holiday": days[di]["holiday"], "promo_day": days[di]["promo_day"]}
    for task in ("day_ahead", "next15"):
        p, lo, hi = fc.forecast_day(bundle, task, days, di)
        base = {b: [fc.baseline(task, b, days, di, j) for j in range(fc.SLOTS_PER_DAY)]
                for b in fc.BASELINES[task]}
        out[task] = {"point": p, "low": lo, "high": hi, "baselines": base,
                     "mae": fc.errors(days[di]["arrivals"], p)["mae"],
                     "baseline_mae": {b: fc.errors(days[di]["arrivals"], v)["mae"] for b, v in base.items()}}
    return out


def _next15_whatif(args):
    """Next-15-minute forecast at one slot, optionally with a counted 'last 15 minutes'."""
    days, bundle = STATE["days"], STATE["bundle"]
    di, j = int(args["day"]), int(args["slot"])
    x = fc.features_next15(days, di, j)
    if args.get("last15") is not None:
        x[fc.FEATURES["next15"].index("lag1")] = float(args["last15"])
    m = bundle["tasks"]["next15"]
    return {"point": max(0.0, fc.predict_one(m["point"], x)), "low": max(0.0, fc.predict_one(m["low"], x)),
            "high": max(0.0, fc.predict_one(m["high"], x)), "actual": days[di]["arrivals"][j],
            "lag1_actual": fc._series_before(days, di, j, 1)}


def _future_day(args):
    """Forecast a made-up day just after the data ends, with chosen calendar and weather."""
    days, bundle = STATE["days"], STATE["bundle"]
    weekday = int(args["weekday"])
    last = days[-1]["weekday"]
    offset = (weekday - last - 1) % 7          # 0..6 days after the last day of data
    di = len(days) + offset
    temp = args.get("temp_c") or [days[-7 + offset]["temp_c"][j] for j in range(fc.SLOTS_PER_DAY)]
    rain = args.get("rain_mm") or [0.0] * fc.SLOTS_PER_DAY
    day = {"weekday": weekday, "holiday": int(args.get("holiday", 0)), "promo_day": int(args.get("promo_day", 0)),
           "school_day": int(args.get("school_day", 1 if weekday < 5 else 0)),
           "minute": days[0]["minute"], "temp_c": temp, "rain_mm": rain}
    p, lo, hi = fc.forecast_day(bundle, "day_ahead", days, di, day=day)
    return {"point": p, "low": lo, "high": hi, "minute": days[0]["minute"], "days_after_data": offset + 1}


def _plan(args):
    rates = [v * 4 for v in args["demand"]]
    costs = args.get("costs") or {}
    res = planner.plan_day(rates, max_staff=int(args.get("max_staff", 7)), reps=int(args.get("reps", 6)),
                           costs=costs)
    out = {"plan": res["plan"], "tried": res["tried"], "shifts": [s[0] for s in planner.SHIFTS]}
    fixed = planner.flat_plan(int(args.get("rota_takers", 1)), int(args.get("rota_crew", 4)))
    if args.get("actual") is not None:
        out["check"] = {"plan": planner.evaluate(res["plan"], counts=args["actual"], reps=10, costs=costs),
                        "rota": planner.evaluate(fixed, counts=args["actual"], reps=10, costs=costs)}
    else:
        out["check"] = {"plan": planner.evaluate(res["plan"], rates_per_hour=rates, reps=10, costs=costs),
                        "rota": planner.evaluate(fixed, rates_per_hour=rates, reps=10, costs=costs)}
    return out


def _gap(args):
    return {"gap_wait_min": sim.gap_wait_from_flow(float(args["vehicles_per_hour"]),
                                                   float(args.get("critical_gap_s", 5.0)))}


HANDLERS = {"load": _load, "compare": _compare, "replay": _replay, "day": _day, "next15": _next15_whatif,
            "future_day": _future_day, "plan": _plan, "gap": _gap}


def call(name, args_json="{}"):
    try:
        return json.dumps({"ok": HANDLERS[name](json.loads(args_json))})
    except Exception as exc:  # reported to the page instead of crashing the worker
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
