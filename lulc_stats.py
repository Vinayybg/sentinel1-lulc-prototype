"""Is a change real? Trend test and noise band for the 12 monthly class shares."""
import math
import numpy as np, pandas as pd
from lulc_model import CLASSES


def _mann_kendall(y):
    n = len(y)
    s = sum(float(np.sign(y[j] - y[i])) for i in range(n - 1) for j in range(i + 1, n))
    _, counts = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - sum(c * (c - 1) * (2 * c + 5) for c in counts)) / 18.0
    if var <= 0 or s == 0:
        return 0.0, 1.0
    z = (s - np.sign(s)) / math.sqrt(var)
    return float(z), float(math.erfc(abs(z) / math.sqrt(2)))


def _theil_sen(y):
    n = len(y)
    return float(np.median([(y[j] - y[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]))


def trend_table(P, min_pp=0.5, alpha=0.05):
    """P: DataFrame (months x classes) of class shares in percent.
    For every class that occurs: Theil-Sen trend over the whole period, the scatter of the monthly values around
    that trend (noise), a Mann-Kendall test, and a verdict. A change counts as a trend only if the test is
    significant AND the trend change over the period is larger than twice the noise and at least min_pp points.
    Note: 12 months cannot separate a real trend from one season-long swing; treat 'trend' as 'worth checking'."""
    rows = []
    for c in CLASSES:
        y = P[c].dropna().values.astype(float)
        if len(y) < 6 or y.max() < 0.05:
            continue
        t = np.arange(len(y))
        slope = _theil_sen(y)
        line = np.median(y - slope * t) + slope * t
        sigma = float((y - line).std(ddof=1))
        z, p = _mann_kendall(y)
        change = slope * (len(y) - 1)
        need = max(min_pp, 2 * sigma)
        verdict = 'trend' if (p < alpha and abs(change) >= need) else 'within noise'
        rows.append((c, round(float(y[-1] - y[0]), 2), round(change, 2), round(sigma, 2), round(need, 2),
                     round(p, 3), verdict))
    return pd.DataFrame(rows, columns=['class', 'first_to_last_pp', 'trend_change_pp', 'monthly_noise_pp',
                                       'needed_to_count_pp', 'p_value', 'verdict']).set_index('class')
