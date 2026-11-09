import numpy as np, pandas as pd

ref = np.load('cache/ref_singapore_2020.npy')
maps = np.load('cache/maps_singapore_2020.npy')
to6 = np.array([0, 1, 2, 2, 2, 2, 3, 4, 5] + [255] * 247, np.uint8)
sel = ref < 5


def smooth(M):
    out = np.empty_like(M)
    for k in range(len(M)):
        w = M[max(0, k - 1):k + 2]
        cnt = np.stack([(w == c).sum(0) for c in range(9)]).astype(np.float32)
        for c in range(9):
            cnt[c] += 0.5 * (M[k] == c)
        out[k] = cnt.argmax(0).astype(np.uint8)
        out[k][M[k] == 255] = 255
    return out


def noise(M):
    flip, net = [], []
    for k in range(len(M) - 1):
        both = (M[k] != 255) & (M[k + 1] != 255)
        flip.append(100 * ((M[k] != M[k + 1]) & both).sum() / both.sum())
        a = np.array([100 * (M[k] == c).sum() / (M[k] != 255).sum() for c in range(9)])
        b = np.array([100 * (M[k + 1] == c).sum() / (M[k + 1] != 255).sum() for c in range(9)])
        net.append(0.5 * np.abs(b - a).sum())
    return np.mean(flip), np.mean(net)


def score(M):
    P = to6[M]
    conf = np.zeros((5, 6), np.int64)
    for k in range(len(M)):
        ok = sel & (P[k] != 255)
        idx = ref[ok].astype(np.int64) * 6 + P[k][ok]
        conf += np.bincount(idx, minlength=30).reshape(5, 6)
    tp = np.diag(conf[:, :5]).astype(float)
    sup = conf.sum(1).astype(float)
    pred = conf[:, :5].sum(0).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = tp / (sup + pred - tp)
    iou[sup == 0] = np.nan
    tp4 = conf[0, 0] + conf[1:3, 1:3].sum() + conf[3, 3]
    return tp.sum() / conf.sum(), np.nanmean(iou), tp4 / conf.sum()


rows = []
for name, M in [('raw monthly maps', maps), ('3-month majority vote', smooth(maps))]:
    f, n = noise(M)
    oa, miou, oa4 = score(M)
    rows.append([name, f, n, f / n, oa, miou, oa4])
df = pd.DataFrame(rows, columns=['version', 'pixels_changing_class_%', 'net_change_pp', 'gross/net_ratio',
                                 'OA_5cls', 'mIoU_5cls', 'OA_merged'])
pd.set_option('display.width', 200)
print(df.round(3).to_string(index=False))
print()
print('Reading: the true land cover is stable in 2020, so the pixels that change class from one month')
print('to the next are noise. gross/net shows how much larger the pixel-level change is than the net change.')
