import numpy as np
import fetch_s1, lulc_model

# Same Singapore area, same independent reference (URA + OSM), same scoring as
# evaluate_singapore.py -- run on 2020 maps and on freshly built 2025 maps.
ref = np.load('cache/ref_singapore_2020.npy')
to6 = np.array([0, 1, 2, 2, 2, 2, 3, 4, 5] + [255] * 247, np.uint8)
NAMES = ['water', 'trees', 'low_veg', 'built', 'bare']


def score(maps):
    pooled = np.zeros((5, 6), np.int64)
    oa_m, miou_m = [], []
    for k in range(maps.shape[0]):
        p = to6[maps[k]]
        ok = p != 255
        conf = np.zeros((5, 6), np.int64)
        for i in range(5):
            conf[i] = np.bincount(p[(ref == i) & ok], minlength=6)[:6]
        tp = np.diag(conf[:, :5]).astype(float)
        sup = conf.sum(1).astype(float)
        pred = conf[:, :5].sum(0).astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = tp / (sup + pred - tp)
        iou[sup == 0] = np.nan
        oa_m.append(tp.sum() / conf.sum())
        miou_m.append(np.nanmean(iou))
        pooled += conf
    tp = np.diag(pooled[:, :5]).astype(float)
    sup = pooled.sum(1).astype(float)
    pred = pooled[:, :5].sum(0).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = tp / (sup + pred - tp)
    return np.mean(oa_m), np.mean(miou_m), np.std(oa_m), iou


m20 = np.load('cache/maps_singapore_2020.npy')
print('Fetching 2025 composites (Dec 2024 - Jan 2026) from Earth Engine ...')
series = fetch_s1.fetch_series(1.3521, 103.8198, 2025, 1)
model = lulc_model.load_model()
m25 = lulc_model.predict_series(model, series, log=lambda *a: None)
np.save('cache/maps_singapore_2025.npy', m25)

print()
print('Singapore, same reference, same scoring (strict = exact-class reference pixels only)')
print('year   OA_strict  OA_std  mIoU_strict  valid%   ' + '  '.join(f'IoU_{n}' for n in NAMES))
for name, m in (('2020', m20), ('2025', m25)):
    oa, miou, sd, iou = score(m)
    print(f'{name}   {oa:8.3f}  {sd:6.4f}  {miou:10.3f}  {100 * (m != 255).mean():6.1f}   ' +
          '  '.join(f'{v:8.3f}' for v in iou))
print()
print('Share of all predicted pixels per class, % (9 classes):')
print('class     ' + ' '.join(f'{c[:6]:>7}' for c in lulc_model.CLASSES))
for name, m in (('2020', m20), ('2025', m25)):
    v = m[m != 255]
    print(f'{name}      ' + ' '.join(f'{100 * (v == c).mean():7.1f}' for c in range(9)))
print()
print('Saved cache/maps_singapore_2025.npy')
