"""SAR signature analysis on the TEST set: how different do the classes look in Sentinel-1 backscatter?
Uses the centre month of each 3-month stack (VV, VH in dB) and the consensus labels. No model needed (about 5 minutes).
Outputs (in <data>/results_sar): class_signature_stats.csv, overlap_2d.csv, overlap_vv.csv, overlap_vh.csv,
pair_overlap_vs_confusion.csv (if confusion_unet_pre.csv exists), sar_signatures.png"""
import argparse, os, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lulc_model import CLASSES

LO, HI, STEP = -35.0, 5.0, 0.5
NBIN = int((HI - LO) / STEP)                     # 80 bins per polarisation
CENT = LO + STEP * (np.arange(NBIN) + 0.5)
COL = ['#2b6cb0', '#1f6b2e', '#8bd35f', '#38b2ac', '#ecc94b', '#a3a43a', '#e53e3e', '#d2b48c', '#a0aec0']
ap = argparse.ArgumentParser()
ap.add_argument('--data', default=r'C:\Users\Vinay\Desktop\lulc-eval')
a = ap.parse_args()
out = os.path.join(a.data, 'results_sar')
os.makedirs(out, exist_ok=True)
Xm = np.load(os.path.join(a.data, 'X_test.npy'), mmap_mode='r')
Ym = np.load(os.path.join(a.data, 'Y_test.npy'), mmap_mode='r')
vi = np.where(np.load(os.path.join(a.data, 'meta_test.npz'))['valid'])[0]
H = np.zeros(9 * NBIN * NBIN, np.int64)          # 2-D histogram (VV, VH) per class
ts_vv, ts_vh, tn = np.zeros(9), np.zeros(9), np.zeros(9)   # temporal std over the 3 months
t0 = time.time()
for s in range(0, len(vi), 32):
    ii = vi[s:s + 32]
    x = np.asarray(Xm[ii], dtype=np.float32)      # (B, 6, 256, 256)
    y = np.asarray(Ym[ii][:, 0]).astype(np.int64)
    ok = (y < 9) & np.isfinite(x).all(1)
    vv, vh = x[:, 2], x[:, 3]
    bv = np.clip(((vv - LO) / STEP).astype(int), 0, NBIN - 1)
    bh = np.clip(((vh - LO) / STEP).astype(int), 0, NBIN - 1)
    H += np.bincount((y * NBIN * NBIN + bv * NBIN + bh)[ok], minlength=9 * NBIN * NBIN)
    sv, sh = x[:, [0, 2, 4]].std(1), x[:, [1, 3, 5]].std(1)
    ts_vv += np.bincount(y[ok], weights=sv[ok], minlength=9)[:9]
    ts_vh += np.bincount(y[ok], weights=sh[ok], minlength=9)[:9]
    tn += np.bincount(y[ok], minlength=9)[:9]
    if (s // 32) % 20 == 0:
        print(f'{s + len(ii)}/{len(vi)} | {(time.time() - t0) / 60:.1f} min', flush=True)
H = H.reshape(9, NBIN, NBIN).astype(np.float64)
n = H.sum((1, 2))


def q(p, quant):
    c = np.cumsum(p)
    return CENT[min(int(np.searchsorted(c, quant * c[-1])), NBIN - 1)]


rows = []
for i, c in enumerate(CLASSES):
    if n[i] < 1000:
        continue
    pv, ph = H[i].sum(1), H[i].sum(0)
    r = dict(cls=c, pixels=int(n[i]))
    for nm, p in (('VV', pv), ('VH', ph)):
        m = (p * CENT).sum() / p.sum()
        r.update({f'{nm}_mean_dB': round(m, 2), f'{nm}_std_dB': round(np.sqrt((p * (CENT - m) ** 2).sum() / p.sum()), 2),
                  f'{nm}_p5': q(p, .05), f'{nm}_median': q(p, .5), f'{nm}_p95': q(p, .95)})
    r['VH-VV_mean_dB'] = round(r['VH_mean_dB'] - r['VV_mean_dB'], 2)
    r['temporal_std_VV_dB'] = round(ts_vv[i] / n[i], 2)
    r['temporal_std_VH_dB'] = round(ts_vh[i] / n[i], 2)
    rows.append(r)
stats = pd.DataFrame(rows).set_index('cls')
stats.to_csv(os.path.join(out, 'class_signature_stats.csv'))
print('\n', stats.to_string())


def overlap(P):
    P = P / np.maximum(P.sum(1, keepdims=True), 1)
    return np.array([[np.minimum(P[i], P[j]).sum() for j in range(9)] for i in range(9)])


O2 = overlap(H.reshape(9, -1))
OV, OH = overlap(H.sum(2)), overlap(H.sum(1))
for nm, O in (('2d', O2), ('vv', OV), ('vh', OH)):
    pd.DataFrame(O.round(3), index=CLASSES, columns=CLASSES).to_csv(os.path.join(out, f'overlap_{nm}.csv'))
print('\nOverlap of (VV,VH) distributions (0 = separable, 1 = identical):')
print(pd.DataFrame(O2.round(2), index=CLASSES, columns=CLASSES).to_string())
# ---- overlap vs confusion
cf = os.path.join(a.data, 'confusion_unet_pre.csv')
rho = None
if os.path.exists(cf):
    C = pd.read_csv(cf, index_col=0).values.astype(float)
    pr = []
    for i in range(9):
        for j in range(i + 1, 9):
            if n[i] < 1000 or n[j] < 1000:
                continue
            conf = (C[i, j] + C[j, i]) / max(C[i].sum() + C[j].sum(), 1)
            pr.append(dict(pair=f'{CLASSES[i]} / {CLASSES[j]}', overlap_2d=round(O2[i, j], 3), confusion_rate_pct=round(100 * conf, 2)))
    pr = pd.DataFrame(pr).sort_values('confusion_rate_pct', ascending=False)
    pr.to_csv(os.path.join(out, 'pair_overlap_vs_confusion.csv'), index=False)
    rho = pr[['overlap_2d', 'confusion_rate_pct']].corr(method='spearman').iloc[0, 1]
    print(f'\nPairs ranked by confusion (Spearman rank correlation overlap vs confusion = {rho:.2f}):')
    print(pr.head(12).to_string(index=False))
# ---- figure
fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
for i, c in enumerate(CLASSES):
    if n[i] < 1000:
        continue
    ax[0, 0].plot(CENT[:-1], (H[i].sum(1) / n[i] / STEP)[:-1], color=COL[i], label=c)   # last bin = values clipped at +5 dB
    ax[0, 1].plot(CENT[:-1], (H[i].sum(0) / n[i] / STEP)[:-1], color=COL[i], label=c)
ax[0, 0].set_title('VV backscatter per class (test set, centre month; data clipped at +5 dB)'); ax[0, 1].set_title('VH backscatter per class')
for k in (0, 1):
    ax[0, k].set_xlabel('dB'); ax[0, k].set_ylabel('density'); ax[0, k].set_xlim(-30, 5); ax[0, k].grid(alpha=.3)
ax[0, 0].legend(ncol=3, fontsize=8)
im = ax[1, 0].imshow(O2, cmap='magma_r', vmin=0, vmax=1)
ax[1, 0].set_xticks(range(9)); ax[1, 0].set_xticklabels(CLASSES, rotation=60); ax[1, 0].set_yticks(range(9)); ax[1, 0].set_yticklabels(CLASSES)
for i in range(9):
    for j in range(9):
        ax[1, 0].text(j, i, f'{O2[i, j]:.2f}', ha='center', va='center', fontsize=7, color='white' if O2[i, j] > .5 else 'black')
ax[1, 0].set_title('Overlap of the (VV, VH) distributions\n(1 = indistinguishable)')
fig.colorbar(im, ax=ax[1, 0], fraction=.046)
if rho is not None:
    ax[1, 1].scatter(pr.overlap_2d, pr.confusion_rate_pct, color='#2b6cb0')
    for _, r in pr.head(6).iterrows():
        ax[1, 1].annotate(r.pair, (r.overlap_2d, r.confusion_rate_pct), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax[1, 1].set_xlabel('SAR distribution overlap'); ax[1, 1].set_ylabel('confusion rate of the U-Net (% of the two classes)')
    ax[1, 1].set_title(f'Overlap vs confusion (Spearman {rho:.2f})'); ax[1, 1].grid(alpha=.3)
else:
    ax[1, 1].axis('off')
    ax[1, 1].text(.5, .5, 'run confusion_local.py first to see\noverlap vs confusion', ha='center')
fig.tight_layout(); fig.savefig(os.path.join(out, 'sar_signatures.png'), dpi=130)
print('\nsaved in', out)
