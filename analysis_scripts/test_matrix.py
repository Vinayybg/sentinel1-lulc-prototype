"""Random-location test matrix: runs the prototype for very different places and prints one summary line each.
Usage:  python test_matrix.py            (all, about 0.2 % Earth Engine quota, 10-20 min)
        python test_matrix.py Sahara     (one place)"""
import sys, time
import numpy as np
from fetch_s1 import fetch_series, LulcError
from lulc_model import load_model, predict_series, CLASSES
from lulc_report import report

PLACES = [  # name, lat, lon, start year, start month, what we expect
    ('Farmland Iowa', 42.00, -93.50, 2020, 1, 'crops/grass, some trees'),
    ('Amazon forest', -5.00, -62.00, 2020, 1, 'trees dominant'),
    ('Sahara desert', 23.00, 12.00, 2020, 1, 'bare dominant'),
    ('Mekong delta', 10.00, 105.80, 2020, 1, 'water + crops + built'),
    ('Sydney', -33.87, 151.21, 2020, 1, 'built + water + trees'),
    ('Lapland', 68.00, 25.50, 2020, 1, 'trees, shrub, snow in winter'),
]


def main():
    only = sys.argv[1].lower() if len(sys.argv) > 1 else None
    model = load_model('unet_pre')
    for name, lat, lon, y, m, expect in PLACES:
        if only and only not in name.lower():
            continue
        t0 = time.time()
        try:
            series = fetch_series(lat, lon, y, m, log=lambda s: None)
            months = int(sum(np.isfinite(s[0]).any() for s in series))
            maps = predict_series(model, series, log=lambda s: None)
            pct, step, fl = report(maps, y, m, f'outputs/matrix_{name.replace(" ", "_")}', f'{name}')
            top = pct[CLASSES].mean().sort_values(ascending=False).head(3)
            share = ', '.join(f'{k} {v:.0f}%' for k, v in top.items())
            print(f'{name:14s} OK  {time.time() - t0:4.0f} s | months with data {months}/14 | top: {share} | expected: {expect}')
        except LulcError as e:
            print(f'{name:14s} MESSAGE: {e}')
        except Exception as e:
            print(f'{name:14s} CRASH: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
