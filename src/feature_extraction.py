"""
feature_extraction.py
----------------------
Turns a raw stream of (key, event_type, timestamp) tuples into the
behavioural-biometric feature vector used by the Random Forest classifier:

    - Dwell Time   (DT_i = R_i - P_i)
    - Flight Time  (FT_ij = P_j - R_i)
    - DEFT         (Flight Time combined with QWERTY key-distance)
    - error_rate   (proportion of backspace / delete presses)

This corresponds to Section V.A / V.B ("Feature Extraction" and "Distance
Enhanced Flight Time") of the paper.
"""

import statistics
from typing import List, Tuple, Dict

from utils import key_distance

BACKSPACE_KEYS = {"key.backspace", "key.delete"}


def extract_features(events: List[Tuple[str, str, float]]) -> Dict[str, float]:
    """
    events: list of (key, 'press'|'release', timestamp) in chronological order.

    Returns a dict of the aggregate features for one typing sample.
    """
    press_times = {}
    dwell_times = []
    flight_times = []
    deft_values = []
    error_count = 0
    total_keys = 0

    ordered_presses = []  # (key, press_time) in order, used for flight time

    for key, event_type, t in events:
        if event_type == "press":
            total_keys += 1
            if key.lower() in BACKSPACE_KEYS:
                error_count += 1
            press_times[key] = t
            ordered_presses.append((key, t))
        elif event_type == "release":
            if key in press_times:
                dwell_times.append(t - press_times[key])

    # Flight time / DEFT between successive key presses
    for idx in range(1, len(ordered_presses)):
        prev_key, prev_t = ordered_presses[idx - 1]
        cur_key, cur_t = ordered_presses[idx]
        ft = cur_t - prev_t
        flight_times.append(ft)
        dist = key_distance(prev_key, cur_key)
        # DEFT: normalise flight time by physical distance travelled
        # (guard against division by zero for repeated keys)
        deft_values.append(ft / dist if dist > 0 else ft)

    def safe_mean(xs):
        return statistics.mean(xs) if xs else 0.0

    def safe_std(xs):
        return statistics.stdev(xs) if len(xs) > 1 else 0.0

    return {
        "avg_dwell_time": round(safe_mean(dwell_times), 4),
        "std_dwell_time": round(safe_std(dwell_times), 4),
        "avg_flight_time": round(safe_mean(flight_times), 4),
        "std_flight_time": round(safe_std(flight_times), 4),
        "avg_deft": round(safe_mean(deft_values), 4),
        "error_rate": round(error_count / total_keys, 4) if total_keys else 0.0,
    }
