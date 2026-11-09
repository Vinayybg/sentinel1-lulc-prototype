"""Command-line version of the prototype.
Example:  python run_lulc.py --lat 1.3521 --lon 103.8198 --year 2020 --month 1
Options:  --model unet_pre|unet_scratch|deeplab_pre|deeplab_scratch|fpn_r50_pre|ensemble   --smooth"""
import argparse, os, sys
import numpy as np
from fetch_s1 import fetch_series, LulcError
from lulc_model import load_model, predict_series, available_models, DEFAULT_MODEL, MODELS
from lulc_report import report
from lulc_smooth import smooth_maps


def main():
    ap = argparse.ArgumentParser(description='Sentinel-1 LULC: 12 monthly maps for a location')
    ap.add_argument('--lat', type=float, required=True)
    ap.add_argument('--lon', type=float, required=True)
    ap.add_argument('--year', type=int, required=True, help='start year of the first map')
    ap.add_argument('--month', type=int, required=True, help='start month (1-12) of the first map')
    ap.add_argument('--model', default=DEFAULT_MODEL, choices=list(MODELS) + ['ensemble'])
    ap.add_argument('--smooth', action='store_true', help='3-month majority vote on the maps')
    a = ap.parse_args()
    try:
        if a.model not in available_models():
            raise LulcError(f'Weights for "{a.model}" not found in weights/. See README, section "Weights".')
        series = fetch_series(a.lat, a.lon, a.year, a.month)
        maps = predict_series(load_model(a.model), series)
    except LulcError as e:
        print('ERROR:', e)
        sys.exit(1)
    if a.smooth:
        maps = smooth_maps(maps)
    out = f'outputs/{a.lat:.4f}_{a.lon:.4f}_{a.year}-{a.month:02d}_{a.model}' + ('_smoothed' if a.smooth else '')
    pct, step, fl = report(maps, a.year, a.month, out, f'LULC {a.lat:.4f}, {a.lon:.4f} ({a.model})')
    print(pct.to_string(), '\n')
    print(fl.to_string(), '\n')
    print('Results saved in', out)


if __name__ == '__main__':
    main()
