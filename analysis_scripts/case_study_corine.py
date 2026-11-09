"""Second independent case study: European cities against CORINE Land Cover 2018 (100 m, Copernicus).
Radar year 2018 matches the reference year. CORINE is an independent product (visual/semi-automatic
interpretation of optical imagery), not one of the training label sources.
Usage:  python case_study_corine.py                 (all cities, about 0.25 % Earth Engine quota, about 25 min)
        python case_study_corine.py --only Munich   (one city, to try it first)
        python case_study_corine.py --models unet_pre   (skip the slower ensemble)"""
import argparse, os, sys, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from fetch_s1 import fetch_series, grid_for, _init_ee, LulcError, SIZE, RES, BASE
from lulc_model import load_model, predict_series, available_models
from lulc_smooth import smooth_maps

YEAR, MONTH = 2018, 1
CITIES = {  # none of these lies within 80 km of a training place (checked against places_final_FROZEN.csv)
    'Mannheim': (49.4875, 8.466), 'Munich': (48.1374, 11.5755), 'Warsaw': (52.2297, 21.0122),
    'Prague': (50.0755, 14.4378), 'Bucharest': (44.4268, 26.1025), 'Madrid': (40.4168, -3.7038),
    'Stockholm': (59.3293, 18.0686)}
CLASSES = ['water', 'trees', 'low_veg', 'built', 'bare', 'snow']
IGN = 255
# CORINE code -> our merged class (255 = not scored: ambiguous or sub-pixel classes)
CLC = {}
for c in (111, 112, 121, 122, 123, 124, 131, 132, 133): CLC[c] = 3
for c in (211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244, 321, 322, 323, 324): CLC[c] = 2
for c in (311, 312, 313): CLC[c] = 1
for c in (331, 332): CLC[c] = 4
CLC[335] = 5
for c in (511, 512, 521, 522, 523): CLC[c] = 0
LUT = np.full(1024, IGN, np.uint8)
for k, v in CLC.items(): LUT[k] = v
TO6 = np.array([0, 1, 2, 2, 2, 2, 3, 4, 5] + [255] * 247, np.uint8)
COL = {0: '#2b6cb0', 1: '#1f6b2e', 2: '#8bd35f', 3: '#e53e3e', 4: '#d2b48c', 5: '#f7fafc', 255: '#2d2d2d'}
OUT = os.path.join(BASE, 'outputs', 'corine_case_study')


def fetch_reference(name, lat, lon):
    p = os.path.join(BASE, 'cache', f'ref_corine2018_{lat:.4f}_{lon:.4f}.npy')
    if os.path.exists(p):
        return np.load(p)
    _init_ee()
    import ee
    epsg, x0, y0 = grid_for(lat, lon)
    img = ee.Image('COPERNICUS/CORINE/V20/100m/2018').select('landcover')
    r = ee.data.computePixels({'expression': img, 'fileFormat': 'NUMPY_NDARRAY',
        'grid': {'dimensions': {'width': SIZE, 'height': SIZE},
                 'affineTransform': {'scaleX': RES, 'shearX': 0, 'translateX': x0, 'shearY': 0, 'scaleY': -RES, 'translateY': y0},
                 'crsCode': f'EPSG:{epsg}'}})
    code = np.asarray(r['landcover']).astype(np.int64)
    ref = LUT[np.clip(code, 0, 1023)]
    np.save(p, ref)
    return ref


def confusion(ref, maps6):
    c = np.zeros((6, 6), np.int64)
    for m in maps6:
        ok = (ref != IGN) & (m != IGN)
        c += np.bincount(ref[ok].astype(int) * 6 + m[ok].astype(int), minlength=36).reshape(6, 6)
    return c


def scores(c):
    tp = np.diag(c).astype(float)
    sup, pred = c.sum(1), c.sum(0)
    iou = np.where(sup + pred - tp > 0, tp / np.maximum(sup + pred - tp, 1), np.nan)
    rec = np.where(sup > 0, tp / np.maximum(sup, 1), np.nan)
    prec = np.where(pred > 0, tp / np.maximum(pred, 1), np.nan)
    present = sup > 0
    return dict(OA=tp.sum() / c.sum(), mIoU=np.nanmean(iou[present]), iou=iou, recall=rec, precision=prec,
                ref_share=100 * sup / sup.sum(), model_share=100 * pred / pred.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, help='one city name')
    ap.add_argument('--models', default='unet_pre,ensemble')
    a = ap.parse_args()
    models = [m for m in a.models.split(',') if m in available_models()]
    cities = {a.only: CITIES[a.only]} if a.only else CITIES
    os.makedirs(OUT, exist_ok=True)
    rows, rowsc, pooled = [], [], {m: np.zeros((6, 6), np.int64) for m in models}
    for name, (lat, lon) in cities.items():
        print(f'\n=== {name} ({lat}, {lon}) vs CORINE 2018 ===')
        t0 = time.time()
        try:
            series = fetch_series(lat, lon, YEAR, MONTH, log=lambda s: None)
            ref = fetch_reference(name, lat, lon)
        except LulcError as e:
            print('SKIPPED:', e)
            continue
        print(f'data ready in {time.time() - t0:.0f} s; scored share of area: {100 * (ref != IGN).mean():.1f} %')
        maps_by = {}
        for mname in models:
            maps = predict_series(load_model(mname), series)
            m6 = TO6[maps]
            maps_by[mname] = m6
            c = confusion(ref, m6)
            pooled[mname] += c
            s = scores(c)
            row = dict(city=name, model=mname, OA_merged=round(s['OA'], 3), mIoU_merged=round(s['mIoU'], 3))
            for i, k in enumerate(CLASSES):
                row[f'recall_{k}'] = round(s['recall'][i], 3)
                row[f'ref_{k}_%'] = round(s['ref_share'][i], 1)
                row[f'model_{k}_%'] = round(s['model_share'][i], 1)
            rows.append(row)
            print(f"{mname:10s} OA {s['OA']:.3f}  mIoU {s['mIoU']:.3f} | built: reference {s['ref_share'][3]:.1f} % vs model {s['model_share'][3]:.1f} % | "
                  f"trees {s['ref_share'][1]:.1f} vs {s['model_share'][1]:.1f} | low_veg {s['ref_share'][2]:.1f} vs {s['model_share'][2]:.1f}")
        mm = models[0]
        lut = np.zeros((256, 3), np.uint8)
        for k, h in COL.items(): lut[k] = [int(h[i:i + 2], 16) for i in (1, 3, 5)]
        fig, ax = plt.subplots(1, 2, figsize=(12, 6.4))
        ax[0].imshow(lut[ref][::2, ::2]); ax[0].set_title(f'{name}: CORINE 2018 (merged)')
        ax[1].imshow(lut[maps_by[mm][0]][::2, ::2]); ax[1].set_title(f'Model {mm}, {YEAR}-{MONTH:02d} (merged)')
        for x in ax: x.axis('off')
        names = dict(zip(range(6), CLASSES)); names[255] = 'not scored'
        fig.legend(handles=[Patch(color=COL[c], label=n) for c, n in names.items() if c in COL], loc='lower center', ncol=7, frameon=False)
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        fig.savefig(os.path.join(OUT, f'{name}_{mm}.png'), dpi=90)
        plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'corine_case_study_per_city.csv'), index=False)
    prow = []
    for mname, c in pooled.items():
        if c.sum() == 0: continue
        s = scores(c)
        r = dict(model=mname, OA_merged=round(s['OA'], 3), mIoU_merged=round(s['mIoU'], 3))
        for i, k in enumerate(CLASSES):
            r[f'recall_{k}'] = round(s['recall'][i], 3); r[f'precision_{k}'] = round(s['precision'][i], 3)
            r[f'ref_{k}_%'] = round(s['ref_share'][i], 1); r[f'model_{k}_%'] = round(s['model_share'][i], 1)
        prow.append(r)
        pd.DataFrame(c, index=CLASSES, columns=CLASSES).to_csv(os.path.join(OUT, f'confusion_{mname}.csv'))
    pd.DataFrame(prow).to_csv(os.path.join(OUT, 'corine_case_study_pooled.csv'), index=False)
    print('\n=== pooled over all cities ===')
    print(pd.DataFrame(prow).T.to_string())
    print('\nsaved in', OUT)


if __name__ == '__main__':
    main()
