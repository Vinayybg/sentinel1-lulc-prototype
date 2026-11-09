import numpy as np
from lulc_model import CLASSES, NODATA


def smooth_maps(maps):
    """3-month majority vote per pixel (the middle month wins ties). No data (255) stays no data."""
    out = np.empty_like(maps)
    for k in range(len(maps)):
        w = maps[max(0, k - 1):k + 2]
        cnt = np.stack([(w == c).sum(0) for c in range(9)]).astype(np.float32)
        for c in range(9):
            cnt[c] += 0.5 * (maps[k] == c)
        out[k] = cnt.argmax(0).astype(np.uint8)
        out[k][maps[k] == 255] = 255
    return out


def window_mode(maps, min_votes=2):
    """Class that at least `min_votes` of the given months agree on (255 = undecided / no data)."""
    cnt = np.stack([(maps == c).sum(0) for c in range(9)])
    top = cnt.max(0)
    mode = cnt.argmax(0).astype(np.uint8)
    mode[top < min_votes] = NODATA
    return mode


def persistent_change(maps, window=3):
    """Persistent change between the first and the last `window` months.

    A pixel's start class is the class that at least 2 of the first 3 months agree on, its end class
    the same for the last 3 months. A pixel counts as changed only if both are decided and differ.
    One-month flicker therefore does not count as change.
    Returns first, last (class maps), both (bool mask where both are decided)."""
    first = window_mode(maps[:window])
    last = window_mode(maps[-window:])
    both = (first != NODATA) & (last != NODATA)
    return first, last, both


def persistent_tables(maps, labels, window=3):
    """Tables for the persistent change (percentage points of the analysed area).
    summary: raw vs persistent change shares; net: per-class start/end share and net pp;
    transitions: biggest from -> to flows (% of the area)."""
    import pandas as pd
    first, last, both = persistent_change(maps, window)
    n = max(int(both.sum()), 1)
    P0 = np.array([100 * ((first == c) & both).sum() / n for c in range(9)])
    P1 = np.array([100 * ((last == c) & both).sum() / n for c in range(9)])
    left = np.array([100 * ((first == c) & (last != c) & both).sum() / n for c in range(9)])
    entered = np.array([100 * ((last == c) & (first != c) & both).sum() / n for c in range(9)])
    net = pd.DataFrame({'start_%': P0, 'end_%': P1, 'net_change_pp': P1 - P0,
                        'pixels_left_class_%': left, 'pixels_entered_class_%': entered},
                       index=CLASSES).round(2)
    a, b = maps[0], maps[-1]
    raw_both = (a != NODATA) & (b != NODATA)
    raw_changed = 100 * ((a != b) & raw_both).sum() / max(int(raw_both.sum()), 1)
    pers_changed = 100 * ((first != last) & both).sum() / n
    net_turnover = 0.5 * np.abs(P1 - P0).sum()
    summary = pd.DataFrame([{
        'period': f'{labels[0]} -> {labels[-1]}',
        'raw_first_vs_last_month_changed_%': round(raw_changed, 2),
        'persistent_changed_%': round(pers_changed, 2),
        'net_change_%_of_area': round(net_turnover, 2),
        'persistent_to_net_ratio': round(pers_changed / net_turnover, 2) if net_turnover > 0 else float('nan'),
        'area_with_decided_class_%': round(100 * both.mean(), 1)}])
    rows = []
    for i in range(9):
        for j in range(9):
            if i != j:
                v = 100 * ((first == i) & (last == j) & both).sum() / n
                if v > 0:
                    rows.append((CLASSES[i], CLASSES[j], round(v, 2)))
    trans = pd.DataFrame(sorted(rows, key=lambda r: -r[2])[:10], columns=['from', 'to', '%_of_area'])
    return summary, net, trans
