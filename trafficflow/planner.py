"""Staffing planner: choose order takers and kitchen crew for each part of the day.

The day (06:00-23:00) is split into five shifts. For each shift, in order, every
allowed staffing option is simulated car by car against the forecast demand on
the same random customers, and the option with the lowest expected total cost is
kept:

    cost = wages + value of customers' waiting time + lost sales from cars that
           drive away + a penalty for every minute a car waits on the public street

Earlier shifts keep the staffing already chosen, so a queue left over from the
lunch rush is carried into the afternoon. This is a greedy, shift-by-shift
search, not a guaranteed optimum over the whole day.
"""

from __future__ import annotations

from . import sim

DAY_START = 6 * 60
SHIFTS = [
    ("Breakfast", 0, 270),      # 06:00-10:30
    ("Lunch", 270, 480),        # 10:30-14:00
    ("Afternoon", 480, 660),    # 14:00-17:00
    ("Dinner", 660, 870),       # 17:00-20:30
    ("Late", 870, 1020),        # 20:30-23:00
]
DEFAULT_COSTS = {
    "wage_per_hour": 16.0,          # per order taker or crew member
    "wait_value_per_min": 0.30,     # what a minute of a customer's waiting costs the business
    "lost_car_value": 6.0,          # lost margin when a car drives away
    "street_value_per_min": 1.0,    # penalty per car-minute queued on the public street
}


def options(max_staff, max_takers=2, max_crew=6):
    return [(t, c) for t in range(1, max_takers + 1) for c in range(1, max_crew + 1) if t + c <= max_staff]


def schedule(plan):
    """[(takers, crew), ...] per shift -> sim schedule lists."""
    takers = [[SHIFTS[k][1], plan[k][0]] for k in range(len(plan))]
    crew = [[SHIFTS[k][1], plan[k][1]] for k in range(len(plan))]
    return takers, crew


def labour_cost(plan, costs):
    total = 0.0
    for (name, a, b), (t, c) in zip(SHIFTS, plan):
        total += (t + c) * (b - a) / 60.0 * costs["wage_per_hour"]
    return total


def staff_hours(plan):
    return sum((t + c) * (b - a) / 60.0 for (_, a, b), (t, c) in zip(SHIFTS, plan))


def customer_cost(result, costs):
    return (result["total_wait"] * costs["wait_value_per_min"]
            + result["lost"] * costs["lost_car_value"]
            + result["street_car_minutes"] * costs["street_value_per_min"])


def _site(plan, site):
    takers, crew = schedule(plan)
    return {**(site or {}), "site": "drive_through", "order_takers": takers, "kitchen_crew": crew}


def plan_day(rates_per_hour, max_staff=7, reps=8, seed=100, costs=None, site=None, progress=None,
             max_takers=2, max_crew=6):
    """Greedy shift-by-shift search. rates_per_hour: 68 slot rates (cars/hour)."""
    costs = {**DEFAULT_COSTS, **(costs or {})}
    opts = options(max_staff, max_takers, max_crew)
    if not opts:
        raise ValueError("max_staff must allow at least one order taker and one crew member")
    chosen = []
    tried = []
    for k, (name, a, b) in enumerate(SHIFTS):
        slots = int(b // 15)
        rates = rates_per_hour[:slots]
        car_sets = [sim.make_cars(rates, seed + r) for r in range(reps)]
        scored = []
        for opt in opts:
            plan = chosen + [opt]
            params = _site(plan, site)
            total = 0.0
            for cars in car_sets:
                res = sim.simulate(params, cars, b)
                total += customer_cost(res, costs)
            score = labour_cost(plan, costs) + total / reps
            scored.append((score, opt[0] + opt[1], opt))
        scored.sort()
        chosen.append(scored[0][2])
        tried.append({"shift": name, "options": [{"takers": o[0], "crew": o[1], "cost": s} for s, _, o in scored]})
        if progress:
            progress(k + 1, len(SHIFTS))
    return {"plan": chosen, "tried": tried}


def evaluate(plan, counts=None, rates_per_hour=None, reps=20, seed=5000, costs=None, site=None):
    """Score a full-day plan on actual counts (or on rates) with fresh random seeds."""
    costs = {**DEFAULT_COSTS, **(costs or {})}
    params = _site(plan, site)
    horizon = SHIFTS[-1][2]
    rows = []
    for r in range(reps):
        cars = sim.cars_from_counts(counts, seed + r) if counts is not None else sim.make_cars(rates_per_hour, seed + r)
        res = sim.simulate(params, cars, horizon)
        rows.append({
            "mean_wait": res["mean_wait"], "p95_wait": res["p95_wait"], "lost": res["lost"],
            "street_car_minutes": res["street_car_minutes"], "customer_cost": customer_cost(res, costs),
        })
    avg = {k: sum(x[k] for x in rows) / reps for k in rows[0]}
    labour = labour_cost(plan, costs)
    avg.update({"labour_cost": labour, "total_cost": labour + avg["customer_cost"],
                "staff_hours": staff_hours(plan), "plan": [list(p) for p in plan]})
    return avg


def flat_plan(takers, crew):
    return [(takers, crew)] * len(SHIFTS)
