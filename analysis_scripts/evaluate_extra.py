import numpy as np, pandas as pd, pyproj, geopandas as gpd
from rasterio import features
from rasterio.transform import Affine
from fetch_s1 import grid_for, SIZE, RES

LAT, LON = 1.3521, 103.8198
epsg, x0, y0 = grid_for(LAT, LON)
tf = Affine(RES, 0, x0, 0, -RES, y0)
tr = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
lo0, la0 = tr.transform(x0 - 500, y0 - SIZE * RES - 500)
lo1, la1 = tr.transform(x0 + SIZE * RES + 500, y0 + 500)

ref = np.load('cache/ref_singapore_2020.npy')
maps = np.load('cache/maps_singapore_2020.npy')
to6 = np.array([0, 1, 2, 2, 2, 2, 3, 4, 5] + [255] * 247, np.uint8)
P = np.stack([to6[m] for m in maps])

# ---- A) merged view: water / vegetation (trees + low_veg) / built / bare
R = np.full(ref.shape, 255, np.uint8)
for a, b in [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3)]:
    R[ref == a] = b
Q = np.full(P.shape, 255, np.uint8)
for a, b in [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 4)]:
    Q[P == a] = b
conf = np.zeros((4, 5), np.int64)
for k in range(12):
    ok = (R != 255) & (Q[k] != 255)
    idx = R[ok].astype(np.int64) * 5 + Q[k][ok]
    conf += np.bincount(idx, minlength=20).reshape(4, 5)
names = ['water', 'vegetation', 'built', 'bare']
tp = np.diag(conf[:, :4]).astype(float)
sup = conf.sum(1).astype(float)
pred = conf[:, :4].sum(0).astype(float)
with np.errstate(divide='ignore', invalid='ignore'):
    iou = tp / (sup + pred - tp)
    f1 = 2 * tp / (sup + pred)
    prec = tp / pred
    rec = tp / sup
for v in (iou, f1, rec):
    v[sup == 0] = np.nan
t = pd.DataFrame({'support_px': sup.astype(int), 'precision': prec, 'recall': rec, 'IoU': iou, 'F1': f1}, index=names).round(3)
print('A) Merged view (trees + low_veg = vegetation), pooled over 12 months')
print(t.to_string())
print('OA %.3f | mIoU %.3f (classes with reference pixels)' % (tp.sum() / conf.sum(), np.nanmean(iou)))

# ---- B) what the model predicts inside each URA "green" zone type
GREEN = ['PARK', 'OPEN SPACE', 'RESERVE SITE', 'CEMETERY', 'SPORTS & RECREATION']
ura = gpd.read_file('reference/ura/MasterPlan2019LandUselayer.geojson', bbox=(lo0, la0, lo1, la1)).to_crs(epsg)
cat = np.zeros(ref.shape, np.uint8)
sh = [(g, GREEN.index(d) + 1) for g, d in zip(ura.geometry.make_valid(), ura['LU_DESC'])
      if d in GREEN and g is not None and not g.is_empty]
features.rasterize(sh, out=cat, transform=tf)
print()
print('B) Model prediction inside URA green zones not resolved by OSM (% of pixels, pooled)')
rows = []
for i, name in enumerate(GREEN, 1):
    m = (cat == i) & (ref == 6)
    if m.sum() == 0:
        continue
    cnt = np.zeros(6)
    for k in range(12):
        p = P[k][m]
        p = p[p != 255]
        cnt += np.bincount(p, minlength=6)[:6]
    rows.append([name, int(m.sum())] + list(100 * cnt / cnt.sum()))
print(pd.DataFrame(rows, columns=['URA zone', 'pixels', 'water', 'trees', 'low_veg', 'built', 'bare', 'snow']).round(1).to_string(index=False))
