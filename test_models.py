"""Quick check of all models on cached Singapore data (no Earth Engine needed).
Run:  python test_models.py"""
import time, numpy as np
import lulc_model as lm

LAT, LON = 1.3521, 103.8198
months = ['2019-12', '2020-01', '2020-02']
series = np.stack([np.load(f'cache/s1_{LAT:.4f}_{LON:.4f}_{m}_1024.npy') for m in months])   # (3,2,H,W)
s6 = np.concatenate([series[0], series[1], series[2]], axis=0)
mean, std = lm.load_norm()
print('input', s6.shape, '| models found:', lm.available_models())

ref = None
for name in lm.available_models():
    model = lm.load_model(name)
    t = time.time()
    m = lm.predict_stack(model, s6, mean, std)
    dt = time.time() - t
    p, v = lm.class_percentages(m)
    print(f'\n{name:16s} {dt:5.1f} s | valid {v:.0f}% | ' + ' '.join(f'{c[:5]}={x:.1f}' for c, x in zip(lm.CLASSES, p)))
    if name == lm.DEFAULT_MODEL:
        ref = m
        t = time.time()
        m_old = lm.predict_stack(model, s6, mean, std, stride=256)
        print(f'  old tiling (stride 256): {time.time() - t:.1f} s')
        both = (m != lm.NODATA) & (m_old != lm.NODATA)
        diff = (m != m_old) & both
        edge = np.zeros(m.shape, bool)
        for k in range(0, m.shape[0], 256):
            edge[max(k - 8, 0):k + 8, :] = True
            edge[:, max(k - 8, 0):k + 8] = True
        print(f'  pixels that differ between old tiling and overlapping windows: {100 * diff.sum() / both.sum():.2f}%')
        print(f'    within 8 px of an old tile border: {100 * diff[edge].sum() / max((both & edge).sum(), 1):.2f}%'
              f' | elsewhere: {100 * diff[~edge].sum() / max((both & ~edge).sum(), 1):.2f}%')
    elif ref is not None:
        both = (m != lm.NODATA) & (ref != lm.NODATA)
        print(f'  agreement with {lm.DEFAULT_MODEL}: {100 * (m == ref)[both].mean():.1f}%')
print('\nTEST DONE')

# ---- ensemble + confidence + trend test ----
import pandas as pd
from lulc_stats import trend_table
if lm.ENSEMBLE in lm.available_models():
    ens = lm.load_model(lm.ENSEMBLE)
    t = time.time()
    m, c = lm.predict_stack(ens, s6, mean, std, return_conf=True)
    p, v = lm.class_percentages(m)
    print(f'\nensemble         {time.time() - t:5.1f} s | ' + ' '.join(f'{cn[:5]}={x:.1f}' for cn, x in zip(lm.CLASSES, p)))
    cf = c.astype(np.float32)
    print(f'  certainty: mean {np.nanmean(cf):.2f}, share below 0.6: {100 * np.nanmean(cf < 0.6):.1f}%')
    both = (m != lm.NODATA) & (ref != lm.NODATA)
    print(f'  agreement with {lm.DEFAULT_MODEL}: {100 * (m == ref)[both].mean():.1f}%')
P = pd.DataFrame({'trees': [34.6, 34.7, 35.3, 36.1, 35.4, 35.0, 34.9, 35.2, 34.9, 35.9, 35.5, 35.3],
                  'built': [58.6, 58.4, 57.7, 57.0, 58.0, 58.4, 58.4, 58.0, 58.3, 57.6, 57.7, 57.8]})
for cn in lm.CLASSES:
    if cn not in P:
        P[cn] = 0.0
print('\ntrend test on Singapore monthly shares:\n', trend_table(P).to_string())
print('\nTEST 2 DONE')
