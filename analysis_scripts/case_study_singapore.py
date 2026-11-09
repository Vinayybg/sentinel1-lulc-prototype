"""Singapore 2020 case study against an independent reference (URA Master Plan 2019 + OpenStreetMap).
Runs from the cache (no Earth Engine). Usage:  python case_study_singapore.py [model ...]
Without arguments all available models are evaluated; the default model gets the detailed output."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lulc_model as lm
from fetch_s1 import fetch_series
from lulc_smooth import smooth_maps, persistent_tables

LAT, LON, Y, M = 1.3521, 103.8198, 2020, 1
OUT = 'outputs/singapore_case_study'
os.makedirs(OUT, exist_ok=True)
ref = np.load('cache/ref_singapore_2020.npy')          # 0 water 1 trees 2 low_veg 3 built 4 bare, 6 green (unspecified), 255 not scored
to6 = np.array([0, 1, 2, 2, 2, 2, 3, 4, 5] + [255] * 247, np.uint8)     # 9 classes -> water,trees,low_veg,built,bare,snow
NAMES = ['water', 'trees', 'low_veg', 'built', 'bare']
LABELS = [f'2020-{k + 1:02d}' for k in range(12)]


def get_maps(name):
    p = f'cache/maps_singapore_2020_{name}.npy'
    if os.path.exists(p):
        return np.load(p)
    series = fetch_series(LAT, LON, Y, M, log=lambda *a: None)
    maps = lm.predict_series(lm.load_model(name), series, log=lambda *a: None)
    np.save(p, maps)
    return maps


def pooled_conf(maps):
    conf = np.zeros((5, 6), np.int64)
    gok = gall = 0
    for k in range(len(maps)):
        p = to6[maps[k]]
        ok = p != 255
        for i in range(5):
            conf[i] += np.bincount(p[(ref == i) & ok], minlength=6)[:6]
        g = (ref == 6) & ok
        gall += int(g.sum()); gok += int((g & np.isin(p, [1, 2])).sum())
    return conf, gok, gall


def scores(conf, gok, gall):
    tp = np.diag(conf[:, :5]).astype(float)
    sup = conf.sum(1).astype(float)
    pred = conf[:, :5].sum(0).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou, rec, prec = tp / (sup + pred - tp), tp / sup, tp / pred
        f1 = 2 * tp / (sup + pred)
    for v in (iou, rec, f1):
        v[sup == 0] = np.nan
    # merged: trees + low_veg -> vegetation
    mv = np.zeros((4, 5), np.int64)
    mp = [0, 1, 1, 2, 3]
    for i in range(5):
        for j in range(5):
            mv[mp[i], mp[j]] += conf[i, j]
        mv[mp[i], 4] += conf[i, 5] * 0     # snow predictions are counted as errors via the row sums below
    rowsum = np.zeros(4)
    for i in range(5):
        rowsum[mp[i]] += conf[i].sum()
    tpm = np.diag(mv[:, :4]).astype(float)
    predm = mv[:, :4].sum(0).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ioum = tpm / (rowsum + predm - tpm)
    ioum[rowsum == 0] = np.nan
    return dict(OA_strict=tp.sum() / conf.sum(), mIoU_strict=np.nanmean(iou),
                OA_incl_green=(tp.sum() + gok) / (conf.sum() + gall),
                OA_merged=tpm.sum() / conf.sum(), mIoU_merged=np.nanmean(ioum),
                recall_low_veg=rec[2], precision_low_veg=prec[2], recall_trees=rec[1], recall_built=rec[3]), iou, rec, prec, f1, sup


def noise(M_):
    flip, net = [], []
    for k in range(len(M_) - 1):
        both = (M_[k] != 255) & (M_[k + 1] != 255)
        flip.append(100 * ((M_[k] != M_[k + 1]) & both).sum() / both.sum())
        a = np.array([100 * (M_[k] == c).sum() / (M_[k] != 255).sum() for c in range(9)])
        b = np.array([100 * (M_[k + 1] == c).sum() / (M_[k + 1] != 255).sum() for c in range(9)])
        net.append(0.5 * np.abs(b - a).sum())
    return float(np.mean(flip)), float(np.mean(net))


names = sys.argv[1:] or lm.available_models()
rows = []
detail = {}
for name in names:
    maps = get_maps(name)
    conf, gok, gall = pooled_conf(maps)
    sc, iou, rec, prec, f1, sup = scores(conf, gok, gall)
    flip, net = noise(maps)
    flip_s, net_s = noise(smooth_maps(maps))
    summ, pnet, ptrans = persistent_tables(maps, LABELS)
    sc.update({'flip_%_raw': flip, 'flip_%_smoothed': flip_s, 'gross_net_raw': flip / net if net else np.nan,
               'persistent_changed_%': float(summ['persistent_changed_%'].iloc[0]),
               'raw_first_last_changed_%': float(summ['raw_first_vs_last_month_changed_%'].iloc[0]),
               'net_change_%': float(summ['net_change_%_of_area'].iloc[0])})
    sc['model'] = name
    rows.append(sc)
    detail[name] = (conf, iou, rec, prec, f1, sup, maps, summ, pnet, ptrans)

df = pd.DataFrame(rows).set_index('model').round(3)
df.to_csv(f'{OUT}/singapore_models_comparison.csv')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('=== Singapore 2020 vs URA + OSM, pooled over 12 months ===')
print(df.T.to_string())

d = lm.DEFAULT_MODEL if lm.DEFAULT_MODEL in detail else names[0]
conf, iou, rec, prec, f1, sup, maps, summ, pnet, ptrans = detail[d]
t = pd.DataFrame({'support_px': sup.astype(int), 'precision': prec, 'recall': rec, 'IoU': iou, 'F1': f1}, index=NAMES).round(3)
t.to_csv(f'{OUT}/per_class_{d}.csv')
print(f'\n=== {d}: per class (5 reference classes, strict) ===')
print(t.to_string())
share = pd.DataFrame({'reference_%': 100 * sup / sup.sum(), 'model_%': 100 * conf[:, :5].sum(0) / conf.sum()}, index=NAMES)
share['diff_pp'] = share['model_%'] - share['reference_%']
print('\nClass share on scored pixels:\n', share.round(2).to_string())
summ.to_csv(f'{OUT}/persistent_change_{d}.csv', index=False)
print('\nPersistent change:\n', summ.to_string(index=False))

cm = conf / np.maximum(conf.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(6)); ax.set_xticklabels(NAMES + ['snow']); ax.set_yticks(range(5)); ax.set_yticklabels(NAMES)
for i in range(5):
    for j in range(6):
        ax.text(j, i, f'{100 * cm[i, j]:.0f}%', ha='center', va='center', color='white' if cm[i, j] > 0.5 else 'black')
ax.set_xlabel('Model prediction'); ax.set_ylabel('Reference (URA + OSM)')
ax.set_title(f'Singapore 2020, {d}: row-normalised confusion matrix')
fig.colorbar(im); fig.tight_layout(); fig.savefig(f'{OUT}/confusion_{d}.png', dpi=110); plt.close(fig)

COL = {0: '#2b6cb0', 1: '#1f6b2e', 2: '#8bd35f', 3: '#e53e3e', 4: '#d2b48c', 5: '#f7fafc', 6: '#b7e4a8', 255: '#2d2d2d'}
lut = np.zeros((256, 3), np.uint8)
for c, h in COL.items():
    lut[c] = [int(h[i:i + 2], 16) for i in (1, 3, 5)]
fig, ax = plt.subplots(1, 2, figsize=(14, 7.4))
ax[0].imshow(lut[ref][::2, ::2]); ax[0].set_title('Reference: URA Master Plan 2019 + OpenStreetMap')
ax[1].imshow(lut[to6[maps[0]]][::2, ::2]); ax[1].set_title(f'Model {d}, 2020-01 (merged to 6 classes)')
for a in ax:
    a.axis('off')
from matplotlib.patches import Patch
nm = {0: 'water', 1: 'trees', 2: 'low_veg', 3: 'built', 4: 'bare', 5: 'snow', 6: 'green (unspecified)', 255: 'not scored'}
fig.legend(handles=[Patch(color=COL[c], label=n) for c, n in nm.items()], loc='lower center', ncol=8, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig(f'{OUT}/reference_vs_model_{d}.png', dpi=100); plt.close(fig)
print(f'\nsaved results in {OUT}/')
