"""Car-by-car (discrete-event) simulation of vehicle queues on a business property.

Every vehicle is followed individually through a line of stages. Each stage has a
number of servers (order takers, a pickup window, an exit lane...) and a fixed
number of waiting spaces in front of it. When the next stage has no free space a
finished vehicle stays where it is and keeps its server busy ("blocking"), so a
full pickup lane holds up the order point and a busy road holds up the window.

Two site types are built in:

* drive_through: order point -> pickup window -> exit onto the road. Food is
  prepared by a kitchen crew in parallel; a car at the window waits for its order.
  When the stack in front of the order point is full, an arriving car either
  drives away or waits on the public street.
* parking_exit: pay gate -> exit onto the road. Cars that cannot get into the
  exit aisle simply wait in their parking space.

Times are in minutes, rates in vehicles per hour. Only the standard library is
used so the same file runs in CPython and in the browser (Pyodide).
"""

from __future__ import annotations

import heapq
import math
import random
from collections import deque

# ---------------------------------------------------------------- inputs

DRIVE_THROUGH_DEFAULTS = {
    "site": "drive_through",
    "order_takers": 1,        # cars that can be ordering at the same time
    "kitchen_crew": 4,        # orders that can be prepared at the same time
    "order_min": 0.6,         # mean minutes to take one order
    "prep_min": 1.5,          # mean minutes for one crew member to prepare one order
    "handoff_min": 0.35,      # mean minutes to pay and hand over at the window
    "stack_spaces": 8,        # waiting spaces in front of the order point
    "pickup_spaces": 4,       # waiting spaces between order point and window
    "exit_spaces": 2,         # waiting spaces between window and the road
    "gap_wait_min": 0.2,      # mean minutes waiting for a safe gap in road traffic
    "merge_min": 0.1,         # minimum minutes to pull out once there is a gap
    "balk_prob": 0.4,         # chance a car drives away when the stack is full
}

PARKING_EXIT_DEFAULTS = {
    "site": "parking_exit",
    "gate_lanes": 1,          # pay/barrier lanes open
    "pay_min": 0.4,           # mean minutes to pay and pass the barrier
    "aisle_spaces": 10,       # waiting spaces in the exit aisle before the gate
    "exit_spaces": 2,         # waiting spaces between the gate and the road
    "gap_wait_min": 0.25,
    "merge_min": 0.1,
}


def schedule_value(value, t):
    """A setting is either a number or a list of [start_minute, number] steps."""
    if isinstance(value, (int, float)):
        return value
    current = value[0][1]
    for start, v in value:
        if t + 1e-9 >= start:
            current = v
        else:
            break
    return current


def schedule_changes(value):
    if isinstance(value, (int, float)):
        return []
    return [start for start, _ in value[1:]]


# ---------------------------------------------------------------- demand

def make_cars(rates, seed, slot_min=15.0, horizon=None):
    """Random arrivals for one run, independent of how the site is operated.

    rates: vehicles/hour for each slot of `slot_min` minutes. Arrivals follow a
    Poisson process whose rate changes by slot. Each car also gets its own random
    work factors (mean 1), so two scenarios given the same cars face exactly the
    same customers ("common random numbers").
    """
    rng = random.Random(seed)
    horizon = horizon if horizon is not None else slot_min * len(rates)
    peak = max(rates) if rates else 0.0
    cars = []
    if peak <= 0:
        return cars
    t = 0.0
    while True:
        t += rng.expovariate(peak / 60.0)
        if t >= horizon:
            break
        slot = min(int(t // slot_min), len(rates) - 1)
        if rng.random() * peak <= rates[slot]:
            cars.append(t)
    attrs = random.Random(seed * 7919 + 17)
    return [
        {
            "arrive": a,
            "f_order": attrs.gammavariate(4.0, 0.25),
            "f_prep": attrs.gammavariate(3.0, 1 / 3.0),
            "f_handoff": attrs.gammavariate(4.0, 0.25),
            "f_merge": attrs.expovariate(1.0),
            "u_balk": attrs.random(),
        }
        for a in cars
    ]


def cars_from_counts(counts, seed, slot_min=15.0):
    """Cars for a day whose arrivals per slot are known (e.g. the actual counts).

    Each slot gets exactly its count, at random times inside the slot.
    """
    rng = random.Random(seed)
    attrs = random.Random(seed * 7919 + 17)
    times = []
    for k, n in enumerate(counts):
        times.extend(sorted(k * slot_min + rng.random() * slot_min for _ in range(int(n))))
    return [
        {
            "arrive": a,
            "f_order": attrs.gammavariate(4.0, 0.25),
            "f_prep": attrs.gammavariate(3.0, 1 / 3.0),
            "f_handoff": attrs.gammavariate(4.0, 0.25),
            "f_merge": attrs.expovariate(1.0),
            "u_balk": attrs.random(),
        }
        for a in times
    ]


def fixed_like(n, seed):
    """n cars already waiting at minute 0 (e.g. counted in a photo), with random work factors."""
    attrs = random.Random(seed * 104729 + 3)
    return [
        {"arrive": 0.0, "f_order": attrs.gammavariate(4.0, 0.25), "f_prep": attrs.gammavariate(3.0, 1 / 3.0),
         "f_handoff": attrs.gammavariate(4.0, 0.25), "f_merge": attrs.expovariate(1.0), "u_balk": 1.0}
        for _ in range(int(n))
    ]


def fixed_cars(times, order=1.0, prep=1.0, handoff=1.0, merge=1.0):
    """Deterministic cars (all work factors fixed) for tests and fluid-model parity."""
    return [
        {"arrive": t, "f_order": order, "f_prep": prep, "f_handoff": handoff, "f_merge": merge, "u_balk": 1.0}
        for t in times
    ]


# ---------------------------------------------------------------- engine

class _Stage:
    def __init__(self, name, servers, spaces, service):
        self.name = name
        self.servers = servers      # number or schedule
        self.spaces = spaces        # waiting spaces in front of the servers
        self.service = service      # f(car, now) -> minutes
        self.queue = deque()        # waiting in the spaces
        self.busy = set()           # being served
        self.blocked = deque()      # finished, waiting for room downstream

    def capacity(self, t):
        return max(0, int(schedule_value(self.servers, t)))

    def occupied(self):
        return len(self.busy) + len(self.blocked)

    def can_accept(self, t):
        free_servers = max(0, self.capacity(t) - self.occupied() - len(self.queue))
        return len(self.queue) < self.spaces + free_servers


class _Kitchen:
    """Parallel order preparation, first come first served."""

    def __init__(self, crew, prep_min):
        self.crew = crew
        self.prep_min = prep_min
        self.free = []  # when each crew member is next free

    def ready_time(self, car, t):
        n = max(0, int(schedule_value(self.crew, t)))
        if n == 0:
            return math.inf
        self.free.sort()
        while len(self.free) > n:   # someone went home: drop the latest-busy slot
            self.free.pop()
        while len(self.free) < n:
            self.free.append(t)
        start = max(t, self.free[0])
        done = start + self.prep_min * car["f_prep"]
        self.free[0] = done
        return done


def _build_stages(p):
    site = p["site"]
    gap = p["gap_wait_min"]
    merge = p["merge_min"]

    def road(car, t):
        g = schedule_value(gap, t)
        if g == math.inf or g is None:
            return math.inf
        return merge + g * car["f_merge"]

    if site == "drive_through":
        kitchen = _Kitchen(p["kitchen_crew"], p["prep_min"])

        def order(car, t):
            return p["order_min"] * car["f_order"]

        def window(car, t):
            return max(0.0, car["ready"] - t) + p["handoff_min"] * car["f_handoff"]

        stages = [
            _Stage("order", p["order_takers"], p["stack_spaces"], order),
            _Stage("window", 1, p["pickup_spaces"], window),
            _Stage("exit", 1, p["exit_spaces"], road),
        ]
        return stages, kitchen, True
    if site == "parking_exit":
        def gate(car, t):
            return p["pay_min"] * car["f_order"]

        stages = [
            _Stage("gate", p["gate_lanes"], p["aisle_spaces"], gate),
            _Stage("exit", 1, p["exit_spaces"], road),
        ]
        return stages, None, False
    raise ValueError(f"unknown site type {site!r}")


def simulate(params, cars, horizon, drain_limit=240.0, trace=False):
    """Run one replication. Returns a dict of results (and per-car records if trace)."""
    defaults = DRIVE_THROUGH_DEFAULTS if params.get("site", "drive_through") == "drive_through" else PARKING_EXIT_DEFAULTS
    p = {**defaults, **params}
    stages, kitchen, can_balk = _build_stages(p)
    n_st = len(stages)
    end_limit = horizon + drain_limit

    recs = [dict(arrive=c["arrive"], into=[None] * n_st, start=[None] * n_st, done=[None] * n_st,
                 leave=None, balked=False) for c in cars]
    work = [dict(c) for c in cars]
    overflow = deque()      # on the street (drive-through) or still in a parking space
    events = []
    seq = 0

    def push(t, kind, a=None, b=None):
        nonlocal seq
        heapq.heappush(events, (t, seq, kind, a, b))
        seq += 1

    for i, c in enumerate(cars):
        push(c["arrive"], "arrive", i)
    for s, st in enumerate(stages):
        for t in schedule_changes(st.servers):
            push(t, "cap", s)
    if kitchen is not None:
        for t in schedule_changes(kitchen.crew):
            push(t, "cap", 0)

    # time-weighted counts for the cars-on-site curve
    on_site = 0
    on_street = 0
    last_t = 0.0
    area_site = 0.0
    area_street = 0.0
    peak_site = 0
    peak_street = 0
    street_minutes = 0.0
    samples = []            # (minute, on_site, on_street)
    next_sample = 0.0

    def advance(t):
        nonlocal last_t, area_site, area_street, street_minutes, next_sample
        while next_sample <= t and next_sample <= end_limit:
            samples.append((next_sample, on_site, on_street))
            next_sample += 1.0
        dt = t - last_t
        if dt > 0:
            area_site += on_site * dt
            area_street += on_street * dt
            if on_street > 0:
                street_minutes += dt
        last_t = t

    def settle(t):
        nonlocal on_site, on_street
        changed = True
        while changed:
            changed = False
            for s in range(n_st - 1, -1, -1):
                st = stages[s]
                while st.queue and st.occupied() < st.capacity(t):
                    i = st.queue.popleft()
                    st.busy.add(i)
                    recs[i]["start"][s] = t
                    dur = st.service(work[i], t)
                    if dur != math.inf:
                        push(t + dur, "done", s, i)
                    changed = True
                if s > 0:
                    prev = stages[s - 1]
                    while prev.blocked and st.can_accept(t):
                        i = prev.blocked.popleft()
                        st.queue.append(i)
                        recs[i]["into"][s] = t
                        changed = True
            while overflow and stages[0].can_accept(t):
                i = overflow.popleft()
                stages[0].queue.append(i)
                recs[i]["into"][0] = t
                if can_balk:   # parking cars were already counted on site
                    on_street -= 1
                    on_site += 1
                changed = True

    while events:
        t, _, kind, a, b = heapq.heappop(events)
        if t > end_limit:
            break
        advance(t)
        if kind == "arrive":
            i = a
            first = stages[0]
            if first.can_accept(t) and not overflow:
                first.queue.append(i)
                recs[i]["into"][0] = t
                on_site += 1
            elif can_balk and work[i]["u_balk"] < p["balk_prob"]:
                recs[i]["balked"] = True
                recs[i]["leave"] = t
            else:
                overflow.append(i)
                if can_balk:
                    on_street += 1
                else:
                    on_site += 1  # parking: still on the property, in a space
        elif kind == "done":
            s, i = a, b
            st = stages[s]
            st.busy.discard(i)
            recs[i]["done"][s] = t
            if kitchen is not None and s == 0:
                work[i]["ready"] = kitchen.ready_time(work[i], t)
            if s == n_st - 1:
                recs[i]["leave"] = t
                on_site -= 1
            else:
                st.blocked.append(i)
        # "cap": a staffing change; settle() starts any work it allows
        settle(t)
        peak_site = max(peak_site, on_site)
        peak_street = max(peak_street, on_street)

    still_there = any(r["leave"] is None for r in recs)
    final_t = end_limit if still_there else max(last_t, horizon)
    advance(final_t)
    return _summarise(recs, samples, horizon, final_t, end_limit, area_site, area_street,
                      peak_site, peak_street, street_minutes, [s.name for s in stages], trace)


def _pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _summarise(recs, samples, horizon, final_t, end_limit, area_site, area_street,
               peak_site, peak_street, street_minutes, stage_names, trace):
    served = [r for r in recs if r["leave"] is not None and not r["balked"]]
    balked = sum(1 for r in recs if r["balked"])
    unfinished = [r for r in recs if r["leave"] is None]
    waits = sorted(r["leave"] - r["arrive"] for r in served)
    # cars still on site when the run stops count with the time they have waited so far
    stuck = [final_t - r["arrive"] for r in unfinished]
    all_waits = sorted(waits + stuck)
    last_leave = max((r["leave"] for r in served), default=0.0)
    out = {
        "arrived": len(recs),
        "served": len(served),
        "lost": balked,
        "unfinished": len(unfinished),
        "served_in_hours": sum(1 for r in served if r["leave"] <= horizon),
        "total_wait": sum(all_waits),
        "mean_wait": sum(all_waits) / len(all_waits) if all_waits else 0.0,
        "p50_wait": _pct(all_waits, 0.5),
        "p95_wait": _pct(all_waits, 0.95),
        "max_wait": all_waits[-1] if all_waits else 0.0,
        "peak_on_site": peak_site,
        "peak_on_street": peak_street,
        "street_minutes": street_minutes,
        "street_car_minutes": area_street,
        "mean_on_site": area_site / max(final_t, 1e-9),
        "clear_after_close": max(0.0, (final_t if unfinished else last_leave) - horizon),
        "never_cleared": bool(unfinished),
        "curve": samples,
        "stages": stage_names,
    }
    if trace:
        out["cars"] = recs
    return out


# ---------------------------------------------------------------- comparisons

METRICS = ["mean_wait", "p95_wait", "lost", "street_car_minutes", "peak_on_site", "served_in_hours",
           "unfinished", "clear_after_close"]


def _band(values):
    v = sorted(values)
    n = len(v)
    mean = sum(v) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return {"mean": mean, "lo": _pct(v, 0.05), "hi": _pct(v, 0.95), "sd": sd,
            "ci": 1.96 * sd / math.sqrt(n) if n > 1 else 0.0}


def gap_wait_from_flow(vehicles_per_hour, critical_gap_s=5.0):
    """Mean minutes a driver waits for a safe gap in random (Poisson) road traffic.

    Classic gap-acceptance result (Adams' delay): with flow q per second and a
    critical gap tc seconds, the mean wait is (e^(q*tc) - 1 - q*tc) / q.
    """
    q = max(0.0, vehicles_per_hour) / 3600.0
    if q == 0:
        return 0.0
    x = min(q * critical_gap_s, 50.0)
    return (math.exp(x) - 1 - x) / q / 60.0


def compare(current, proposed, rates, reps=30, seed=1, slot_min=15.0, initial_queue=0):
    """Run both scenarios on the same random customers `reps` times.

    Returns, for every metric, the spread for each scenario and of the paired
    difference (proposed minus current), plus median cars-on-site curves.
    """
    horizon = slot_min * len(rates)
    runs = {"current": [], "proposed": []}
    for r in range(reps):
        cars = fixed_like(initial_queue, seed + r) + make_cars(rates, seed + r, slot_min)
        runs["current"].append(simulate(current, cars, horizon))
        runs["proposed"].append(simulate(proposed, cars, horizon))
    summary = {}
    for m in METRICS:
        cur = [float(x[m]) for x in runs["current"]]
        pro = [float(x[m]) for x in runs["proposed"]]
        diff = [b - a for a, b in zip(cur, pro)]
        summary[m] = {"current": _band(cur), "proposed": _band(pro), "diff": _band(diff),
                      "better_share": sum(1 for d in diff if d < 0) / reps,
                      "higher_share": sum(1 for d in diff if d > 0) / reps}
    return {
        "reps": reps,
        "horizon": horizon,
        "summary": summary,
        "arrived": _band([float(x["arrived"]) for x in runs["current"]]),
        "curves": {k: _curve_band([x["curve"] for x in v], horizon) for k, v in runs.items()},
    }


def _curve_band(curves, horizon):
    length = int(horizon) + 1
    out = []
    for minute in range(0, length, 1):
        site = sorted((c[minute][1] + c[minute][2]) if minute < len(c) else 0 for c in curves)
        out.append([minute, _pct(site, 0.1), _pct(site, 0.5), _pct(site, 0.9)])
    return out
