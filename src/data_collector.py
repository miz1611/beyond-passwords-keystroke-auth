"""
data_collector.py
------------------
Captures raw key-press / key-release timestamps while a user types a fixed
prompt phrase, and appends one row of (dwell, flight, DEFT) features to a
CSV file, labelled with the user's ID.

This is the "Data Acquisition" stage of the pipeline described in the
paper (Section V).

Usage:
    python data_collector.py --user misbah --label 1 --rounds 10
    python data_collector.py --user imposter1 --label 0 --rounds 10

Requires: pynput  (pip install pynput)
"""

import argparse
import csv
import os
import time

from pynput import keyboard

from feature_extraction import extract_features

PROMPT_PHRASE = "the quick brown fox"
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "keystroke_features.csv")


def capture_one_round(prompt: str):
    """
    Records key-press / key-release events while the user types `prompt`
    once, then returns the raw event list: [(key, 'press'/'release', t), ...]
    """
    events = []
    typed = []
    done = threading_event = None
    finished = {"flag": False}

    print(f"\nType exactly:  {prompt}")
    print("(press Enter when done)\n")

    def on_press(key):
        try:
            k = key.char
        except AttributeError:
            k = "space" if key == keyboard.Key.space else str(key)
        events.append((k, "press", time.time()))

    def on_release(key):
        try:
            k = key.char
        except AttributeError:
            k = "space" if key == keyboard.Key.space else str(key)
        events.append((k, "release", time.time()))
        if key == keyboard.Key.enter:
            finished["flag"] = True
            return False  # stop listener

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="user ID / name")
    parser.add_argument("--label", required=True, type=int,
                         help="1 = genuine user, 0 = imposter")
    parser.add_argument("--rounds", type=int, default=10,
                         help="number of typing samples to collect")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    write_header = not os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "user", "label", "avg_dwell_time", "std_dwell_time",
                "avg_flight_time", "std_flight_time", "avg_deft", "error_rate"
            ])

        for i in range(args.rounds):
            print(f"\n=== Round {i + 1}/{args.rounds} for user '{args.user}' ===")
            events = capture_one_round(PROMPT_PHRASE)
            feats = extract_features(events)
            writer.writerow([
                args.user, args.label,
                feats["avg_dwell_time"], feats["std_dwell_time"],
                feats["avg_flight_time"], feats["std_flight_time"],
                feats["avg_deft"], feats["error_rate"],
            ])
            print(f"Captured: {feats}")

    print(f"\nDone. Features appended to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
