"""Tests for the simulator, forecast and planner. Run: python tests/test_trafficflow.py (or pytest)."""

import json
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from trafficflow import api, forecast as fc, planner, sim  # noqa: E402

DATA = os.path.join(ROOT, "data", "synthetic_drive_through.csv")
MODELS = os.path.join(ROOT, "web", "data", "models.json")


def every(times_per_hour, minutes):
    gap = 60.0 / times_per_hour
    return [i * gap for i in range(int(minutes / gap))]


def parking(**kw):
    p = dict(site="parking_exit", gate_lanes=1, pay_min=1.0, aisle_spaces=10000, exit_spaces=2,
             gap_wait_min=0.0, merge_min=1 / 3)
    p.update(kw)
    return p


def conserved(res):
    return res["arrived"] == res["served"] + res["lost"] + res["unfinished"]


# ---------------------------------------------------------------- parity with the old FlowLab cases

def test_slow_service_matches_fluid_model():
    # FlowLab: arrivals 120/h, one position at 60/h, exit 180/h -> 60 served, 60 queued after an hour
    res = sim.simulate(parking(), sim.fixed_cars(every(120, 60)), 60)
    assert abs(res["served_in_hours"] - 60) <= 1
    assert abs(res["curve"][60][1] - 60) <= 1.5
    assert conserved(res)


def test_parking_exit_matches_fluid_model():
    # FlowLab: arrivals 120/h, exit 90/h -> 90 out, 30 waiting after an hour
    res = sim.simulate(parking(pay_min=0.0, merge_min=60 / 90), sim.fixed_cars(every(120, 60)), 60)
    assert abs(res["served_in_hours"] - 90) <= 1
    assert abs(res["curve"][60][1] - 30) <= 1.5


def test_spare_capacity_means_no_queue():
    res = sim.simulate(parking(pay_min=0.5, merge_min=0.2), sim.fixed_cars(every(60, 60)), 60)
    assert res["served"] == 60
    assert res["peak_on_site"] <= 1
    assert abs(res["mean_wait"] - 0.7) < 1e-9          # every car just pays and merges


def test_initial_queue_drains():
    # FlowLab: 12 cars already waiting, no new arrivals, capacity 60/h -> clear in about 12 minutes
    res = sim.simulate(parking(), sim.fixed_cars([0.0] * 12), 0)
    assert res["served"] == 12
    assert 11.5 <= res["clear_after_close"] <= 12.5


def test_zero_staff_serves_nobody():
    res = sim.simulate(parking(gate_lanes=0), sim.fixed_cars(every(120, 60)), 60, drain_limit=30)
    assert res["served"] == 0 and res["unfinished"] == 120 and conserved(res)


# ---------------------------------------------------------------- what FlowLab could not do

def test_blocked_exit_backs_up_the_whole_drive_through():
    cars = sim.make_cars([60] * 4, seed=3)
    res = sim.simulate({"site": "drive_through", "gap_wait_min": math.inf}, cars, 60, drain_limit=30, trace=True)
    assert res["served"] == 0
    assert res["lost"] > 0                               # the stack filled up, so cars drove away
    assert res["peak_on_street"] > 0                     # ... or waited on the street
    p = sim.DRIVE_THROUGH_DEFAULTS
    # with the exit blocked, at most (exit spaces + 1) cars ever leave the window,
    # so the window and the order point stop too
    past_window = sum(1 for c in res["cars"] if c["done"][1] is not None)
    assert past_window <= p["exit_spaces"] + 1 + 1
    assert conserved(res)


def test_waiting_spaces_are_never_overfilled():
    cars = sim.make_cars([140] * 8, seed=11)
    res = sim.simulate({"site": "drive_through", "kitchen_crew": 2}, cars, 120, trace=True)
    spaces = [sim.DRIVE_THROUGH_DEFAULTS[k] for k in ("stack_spaces", "pickup_spaces", "exit_spaces")]
    events = []
    for c in res["cars"]:
        for s in range(3):
            if c["into"][s] is not None:
                events.append((c["into"][s], 1, s))
                end = c["start"][s] if c["start"][s] is not None else math.inf
                events.append((end, -1, s))
    events.sort(key=lambda e: (e[0], e[1]))             # leave before enter at the same instant
    level = [0, 0, 0]
    for _, d, s in events:
        level[s] += d
        # a car can pass through a space at the instant a server frees up, so allow
        # the servers on top of the waiting spaces
        assert level[s] <= spaces[s] + 1, (s, level)
    assert conserved(res)


def test_every_car_is_accounted_for():
    for seed in range(5):
        cars = sim.make_cars([30, 90, 150, 60], seed=seed)
        for params in ({"site": "drive_through"}, {"site": "drive_through", "order_takers": 2, "kitchen_crew": 6},
                       {"site": "parking_exit", "gate_lanes": 2}):
            assert conserved(sim.simulate(params, cars, 60))


def test_littles_law_holds_in_a_steady_run():
    cars = sim.make_cars([40] * 40, seed=5)            # 10 hours at a steady 40 cars/hour
    res = sim.simulate(parking(pay_min=1.0, merge_min=0.1), cars, 600)
    rate = res["arrived"] / (600 + res["clear_after_close"])
    assert abs(res["mean_on_site"] - rate * res["mean_wait"]) / res["mean_on_site"] < 0.05


def test_same_scenario_twice_gives_zero_difference():
    out = sim.compare({"site": "drive_through"}, {"site": "drive_through"}, [80] * 8, reps=5)
    assert all(v["diff"]["mean"] == 0 for v in out["summary"].values())


def test_more_kitchen_crew_cuts_waiting():
    out = sim.compare({"site": "drive_through", "kitchen_crew": 2}, {"site": "drive_through", "kitchen_crew": 5},
                      [90] * 8, reps=10)
    assert out["summary"]["mean_wait"]["diff"]["mean"] < 0
    assert out["summary"]["mean_wait"]["better_share"] >= 0.9


def test_staffing_schedule_changes_capacity_mid_run():
    cars = sim.fixed_cars(every(120, 120))
    flat = sim.simulate(parking(), cars, 120)
    boosted = sim.simulate(parking(gate_lanes=[[0, 1], [60, 2]]), cars, 120)
    assert boosted["served_in_hours"] > flat["served_in_hours"] + 20


def test_gap_wait_grows_with_road_traffic():
    waits = [sim.gap_wait_from_flow(q) for q in (0, 300, 600, 1200)]
    assert waits[0] == 0 and waits == sorted(waits)


# ---------------------------------------------------------------- forecast

def _load():
    with open(DATA) as f:
        days = fc.load_days(f.read())
    with open(MODELS) as f:
        return days, json.load(f)


def test_features_never_look_at_the_future():
    days, _ = _load()
    di, j = 90, 30
    before_next = fc.features_next15(days, di, j)
    before_day = fc.features_day_ahead(days, di, j)
    saved = list(days[di]["arrivals"])
    days[di]["arrivals"][j:] = [999] * (fc.SLOTS_PER_DAY - j)    # change the slot being predicted and later
    assert fc.features_next15(days, di, j) == before_next
    days[di]["arrivals"][:] = [999] * fc.SLOTS_PER_DAY            # day-ahead may not see any of the day
    assert fc.features_day_ahead(days, di, j) == before_day
    days[di]["arrivals"][:] = saved


def test_exported_models_reproduce_reported_test_scores():
    days, bundle = _load()
    start, stop = bundle["test_days"]
    for task in ("next15", "day_ahead"):
        actual, pred = [], []
        for di in range(start, stop):
            p, lo, hi = fc.forecast_day(bundle, task, days, di)
            actual += days[di]["arrivals"]
            pred += p
            assert all(a <= b <= c for a, b, c in zip(lo, p, hi))
        info = bundle["tasks"][task]
        assert abs(fc.errors(actual, pred)["mae"] - info["test"][info["chosen"]]["mae"]) < 0.01


def test_reported_baselines_are_recomputed_exactly():
    days, bundle = _load()
    start, stop = bundle["test_days"]
    for task in ("next15", "day_ahead"):
        for b in fc.BASELINES[task]:
            actual, pred = [], []
            for di in range(start, stop):
                for j in range(fc.SLOTS_PER_DAY):
                    actual.append(days[di]["arrivals"][j])
                    pred.append(fc.baseline(task, b, days, di, j))
            assert abs(fc.errors(actual, pred)["mae"] - bundle["tasks"][task]["test"][b]["mae"]) < 1e-9


def test_test_weeks_come_after_training_weeks():
    _, bundle = _load()
    s = bundle["split"]
    assert s["train"][1] < s["validation"][0] <= s["validation"][1] < s["test"][0]


# ---------------------------------------------------------------- planner and page API

def test_planner_respects_staff_limit():
    days, _ = _load()
    res = planner.plan_day([v * 4 for v in days[90]["arrivals"]], max_staff=4, reps=2)
    assert len(res["plan"]) == len(planner.SHIFTS)
    assert all(t + c <= 4 and t >= 1 and c >= 1 for t, c in res["plan"])


def test_api_round_trip():
    with open(DATA) as f:
        csv_text = f.read()
    with open(MODELS) as f:
        models = f.read()
    ok = json.loads(api.call("load", json.dumps({"csv": csv_text, "models": models})))
    assert "ok" in ok and ok["ok"]["n_days"] == 98
    day = json.loads(api.call("day", json.dumps({"day": 90})))["ok"]
    assert len(day["day_ahead"]["point"]) == fc.SLOTS_PER_DAY
    fut = json.loads(api.call("future_day", json.dumps({"weekday": 4, "promo_day": 1})))["ok"]
    assert len(fut["point"]) == fc.SLOTS_PER_DAY
    bad = json.loads(api.call("compare", json.dumps({})))
    assert "error" in bad


if __name__ == "__main__":
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL", name, "->", repr(exc))
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
