"""Make the SYNTHETIC drive-through dataset used to train and test the forecast.

Nothing here is measured at a real restaurant. The numbers come from a made-up
process written below (daily rush hours, weekday effects, rain, heat, promotion
days, a holiday, school starting, slow growth and random noise), so the forecast
can be trained and tested honestly on data it has never seen. Accuracy on this
data says nothing about accuracy at a real site.

Output: data/synthetic_drive_through.csv, one row per 15-minute slot, 06:00-23:00,
14 weeks starting Monday 6 July 2026.

Run:  python training/make_data.py
"""

import csv
import datetime as dt
import math
import os
import random

START = dt.date(2026, 7, 6)          # a Monday
WEEKS = 14
OPEN_MIN = 6 * 60
CLOSE_MIN = 23 * 60
SLOT = 15
SCHOOL_STARTS = dt.date(2026, 8, 24)
HOLIDAYS = {dt.date(2026, 9, 7): "Labor Day"}
SEED = 2026

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "synthetic_drive_through.csv")


def bump(x, centre, width, height):
    return height * math.exp(-0.5 * ((x - centre) / width) ** 2)


def base_rate(minute, weekday, school):
    """Cars per hour for an ordinary day at this minute of the day."""
    h = minute / 60.0
    weekend = weekday >= 5
    if weekend:
        rate = bump(h, 9.3, 1.1, 38) + bump(h, 12.6, 1.2, 105) + bump(h, 18.2, 1.3, 80)
    else:
        breakfast = 52 if school else 40
        rate = bump(h, 7.6, 0.9, breakfast) + bump(h, 12.25, 0.85, 100) + bump(h, 18.0, 1.1, 72)
        if school:
            rate += bump(h, 15.4, 0.5, 22)   # after-school snack run
    rate += 12 + bump(h, 21.0, 1.2, 10) - 10 * max(0.0, h - 21.5)
    day_mult = [0.90, 0.92, 0.95, 1.00, 1.15, 1.08, 0.97][weekday]
    return max(3.0, 0.65 * rate * day_mult)


def main():
    rng = random.Random(SEED)
    rows = []
    for d in range(WEEKS * 7):
        day = START + dt.timedelta(days=d)
        weekday = day.weekday()
        school = day >= SCHOOL_STARTS and weekday < 5 and day not in HOLIDAYS
        holiday = day in HOLIDAYS
        promo = (not holiday) and rng.random() < 0.1
        # weather: daily high falls slowly from summer into autumn
        season_high = 31.0 - 9.0 * d / (WEEKS * 7)
        high = season_high + rng.gauss(0, 2.5)
        rainy_day = rng.random() < 0.25
        rain_start = rng.uniform(6, 20)
        rain_len = rng.uniform(1.0, 6.0)
        rain_strength = rng.uniform(0.5, 6.0)
        growth = 1.0 + 0.002 * (d / 7)
        for minute in range(OPEN_MIN, CLOSE_MIN, SLOT):
            h = (minute + SLOT / 2) / 60
            temp = high - 7.0 + 7.0 * math.sin(math.pi * max(0.0, min(1.0, (h - 6) / 14)))
            raining = rainy_day and rain_start <= h < rain_start + rain_len
            rain_mm = round(rain_strength * rng.uniform(0.6, 1.4), 1) if raining else 0.0
            rate = base_rate(minute + SLOT / 2, weekday, school)
            if holiday:
                rate = base_rate(minute + SLOT / 2, 6, False) * 1.1
            if promo and (11 <= h < 14 or 17 <= h < 20):
                rate *= 1.25
            if rain_mm > 0:
                rate *= 1.0 + min(0.3, 0.06 * rain_mm)
            if temp > 29 and 13 <= h < 18:
                rate *= 1.08
            rate *= growth
            expected = rate * SLOT / 60
            noisy = expected * rng.gammavariate(20, 1 / 20)
            # Poisson draw (Knuth for small means is fine here)
            limit = math.exp(-noisy)
            k, prod = 0, rng.random()
            while prod > limit:
                k += 1
                prod *= rng.random()
            rows.append({
                "date": day.isoformat(),
                "weekday": weekday,
                "slot_start": f"{minute // 60:02d}:{minute % 60:02d}",
                "minute_of_day": minute,
                "arrivals": k,
                "temp_c": round(temp, 1),
                "rain_mm": rain_mm,
                "holiday": int(holiday),
                "promo_day": int(promo),
                "school_day": int(school),
            })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
