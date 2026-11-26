"""Does planning staff from the ML forecast actually help? Test it on the 14 test days.

For every test day, four staffing plans are built:
  * fixed rota: 1 order taker + 4 kitchen crew all day (a common flat rota)
  * from "same time last week": the planner run on last week's counts
  * from the ML forecast: the planner run on the day-ahead model's forecast
  * from hindsight: the planner run on the day's actual counts (a best case no
    real forecast can reach; it shows how much is left to gain)

Each plan is then scored by simulating the day's ACTUAL arrivals (synthetic
data) with fresh random seeds that the planner never saw.

Run:  python training/evaluate_planner.py   (about two minutes)
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from trafficflow import forecast as fc, planner  # noqa: E402

DATA = os.path.join(ROOT, "data", "synthetic_drive_through.csv")
MODELS = os.path.join(ROOT, "web", "data", "models.json")
OUT = os.path.join(ROOT, "web", "data", "planner_eval.json")
MAX_STAFF = 7


def demand(bundle, days, di, mode):
    point, _, high = fc.forecast_day(bundle, "day_ahead", days, di)
    if mode == "cautious":   # halfway between the forecast and its high estimate
        return [(p + h) / 2 for p, h in zip(point, high)]
    return point


def choose_mode(bundle, days, val_range):
    """Pick, on the VALIDATION weeks, whether to plan for the forecast or a cautious margin above it."""
    cost = {}
    for mode in ("forecast", "cautious"):
        total = 0.0
        for di in val_range:
            plan = planner.plan_day([v * 4 for v in demand(bundle, days, di, mode)], MAX_STAFF)["plan"]
            total += planner.evaluate(plan, counts=days[di]["arrivals"])["total_cost"]
        cost[mode] = total / len(val_range)
        print(f"validation: planning for '{mode}' costs ${cost[mode]:.0f}/day")
    return min(cost, key=cost.get), cost


def main():
    with open(DATA) as f:
        days = fc.load_days(f.read())
    with open(MODELS) as f:
        bundle = json.load(f)
    start, stop = bundle["test_days"]
    mode, val_costs = choose_mode(bundle, days, range(start - 14, start))
    rows = []
    for di in range(start, stop):
        actual = days[di]["arrivals"]
        ml = demand(bundle, days, di, mode)
        last_week = [float(v) for v in days[di - 7]["arrivals"]]
        plans = {
            "fixed_rota": planner.flat_plan(1, 4),
            "last_week_plan": planner.plan_day([v * 4 for v in last_week], MAX_STAFF)["plan"],
            "ml_plan": planner.plan_day([v * 4 for v in ml], MAX_STAFF)["plan"],
            "hindsight_plan": planner.plan_day([v * 4 for v in actual], MAX_STAFF)["plan"],
        }
        scored = {k: planner.evaluate(p, counts=actual) for k, p in plans.items()}
        rows.append({"date": days[di]["date"], "weekday": days[di]["weekday"], "arrivals": sum(actual), **scored})
        print(days[di]["date"], {k: round(v["total_cost"]) for k, v in scored.items()})
    names = ["fixed_rota", "last_week_plan", "ml_plan", "hindsight_plan"]
    totals = {}
    for k in names:
        totals[k] = {m: sum(r[k][m] for r in rows) / len(rows)
                     for m in ["total_cost", "labour_cost", "customer_cost", "mean_wait", "p95_wait", "lost",
                               "street_car_minutes", "staff_hours"]}
    ml_beats_last_week = sum(1 for r in rows if r["ml_plan"]["total_cost"] < r["last_week_plan"]["total_cost"])
    out = {"days": rows, "average_per_day": totals, "max_staff": MAX_STAFF, "costs": planner.DEFAULT_COSTS,
           "ml_beats_last_week_days": ml_beats_last_week,
           "planning_demand": mode, "validation_costs": val_costs, "n_days": len(rows)}
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("\naverage per day:")
    for k in names:
        t = totals[k]
        print(f"  {k:15s} total ${t['total_cost']:.0f}  labour ${t['labour_cost']:.0f}  "
              f"wait {t['mean_wait']:.2f} min  lost {t['lost']:.1f}  staff-hours {t['staff_hours']:.1f}")
    print(f"ML plan cheaper than last-week plan on {ml_beats_last_week}/{len(rows)} days")


if __name__ == "__main__":
    main()
