"""
utils.py
--------
Helper utilities for the Beyond Passwords keystroke dynamics project.

Contains a QWERTY keyboard layout map (row, column positions for each key)
and a function to compute the physical Euclidean distance between any two
keys. This distance is what feeds the "Distance Enhanced Flight Time"
(DEFT) feature described in the paper.
"""

import math

# Approximate (row, col) grid position of each key on a standard QWERTY
# keyboard. Row 0 = number row, Row 1 = qwerty row, Row 2 = asdf row,
# Row 3 = zxcv row. Columns are offset slightly per row to mimic the
# real physical stagger of a keyboard.
QWERTY_LAYOUT = {
    # number row
    '1': (0, 0), '2': (0, 1), '3': (0, 2), '4': (0, 3), '5': (0, 4),
    '6': (0, 5), '7': (0, 6), '8': (0, 7), '9': (0, 8), '0': (0, 9),
    # top letter row (offset by 0.25)
    'q': (1, 0.25), 'w': (1, 1.25), 'e': (1, 2.25), 'r': (1, 3.25),
    't': (1, 4.25), 'y': (1, 5.25), 'u': (1, 6.25), 'i': (1, 7.25),
    'o': (1, 8.25), 'p': (1, 9.25),
    # home row (offset by 0.5)
    'a': (2, 0.5), 's': (2, 1.5), 'd': (2, 2.5), 'f': (2, 3.5),
    'g': (2, 4.5), 'h': (2, 5.5), 'j': (2, 6.5), 'k': (2, 7.5),
    'l': (2, 8.5),
    # bottom row (offset by 0.75)
    'z': (3, 0.75), 'x': (3, 1.75), 'c': (3, 2.75), 'v': (3, 3.75),
    'b': (3, 4.75), 'n': (3, 5.75), 'm': (3, 6.75),
    # space bar treated as a single wide key centred below the letter rows
    'space': (4, 4.5),
}


def key_distance(key_i: str, key_j: str) -> float:
    """
    Euclidean distance between two keys on the QWERTY grid.

    Falls back to a fixed "average" distance (5.0) for keys not in the
    layout map (punctuation, modifiers, etc.) so the pipeline never
    crashes on an unexpected character.
    """
    key_i = key_i.lower()
    key_j = key_j.lower()

    if key_i not in QWERTY_LAYOUT or key_j not in QWERTY_LAYOUT:
        return 5.0

    r1, c1 = QWERTY_LAYOUT[key_i]
    r2, c2 = QWERTY_LAYOUT[key_j]
    return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)


if __name__ == "__main__":
    # quick sanity check
    print("t -> h distance:", key_distance("t", "h"))
    print("a -> l distance:", key_distance("a", "l"))
