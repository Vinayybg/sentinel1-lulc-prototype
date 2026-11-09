import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ref = np.load('cache/ref_singapore_2020.npy')
maps = np.load('cache/maps_singapore_2020.npy')
to6 = np.array([0, 1, 2, 2, 2, 2, 3, 4, 5] + [255] * 247, np.uint8)
NAMES = ['water', 'trees', 'low_veg', 'built', 'bare']
PRED = NAMES + ['snow']


def metrics(conf):
    tp = np.diag(conf[:, :5]).astype(float)
    sup = conf.sum(1).astype(float)
    pred = conf[:, :5].sum(0).astype(float) + 0.0
    fp = pred - tp
    fn = sup - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = tp / (tp + fp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        prec = tp / pred
        rec = tp / sup
    iou[sup == 0] = np.nan; f1[sup == 0] = np.nan; rec[sup == 0] = np.nan
    return iou, f1, prec, rec, sup


rows, pooled, gok_t, gall_t = [], np.zeros((5, 6), np.int64), 0, 0
for k in range(12):
    p = to6[maps[k]]
    ok = p != 255
    conf = np.zeros((5, 6), np.int64)
    for i in range(5):
        sel = (ref == i) & ok
        conf[i] = np.bincount(p[sel], minlength=6)[:6]
    g = (ref == 6) & ok
    gall, gok = int(g.sum()), int((g & np.isin(p, [1, 2])).sum())
    iou, f1, prec, rec, sup = metrics(conf)
    tp = np.diag(conf[:, :5]).sum()
    row = {'month': f'2020-{k + 1:02d}',
           'OA_strict': tp / conf.sum(),
           'OA_incl_green': (tp + gok) / (conf.sum() + gall),
           'mIoU_strict': np.nanmean(iou)}
    for n, v in zip(NAMES, iou):
        row['IoU_' + n] = v
    for n, v in zip(NAMES, f1):
        row['F1_' + n] = v
    rows.append(row)
    pooled += conf; gok_t += gok; gall_t += gall

df = pd.DataFrame(rows).round(4)
df.to_csv('outputs/singapore_2020-01/evaluation_per_month.csv', index=False)

iou, f1, prec, rec, sup = metrics(pooled)
tp = np.diag(pooled[:, :5]).sum()
pd.set_option('display.width', 200)
print('Per month (strict = only pixels where the reference names an exact class):')
print(df[['month', 'OA_strict', 'OA_incl_green', 'mIoU_strict']].to_string(index=False))
print()
print('Mean over 12 months:  OA %.3f | OA incl. green %.3f | mIoU %.3f' % (
    df.OA_strict.mean(), df.OA_incl_green.mean(), df.mIoU_strict.mean()))
print('Std over 12 months :  OA %.4f | mIoU %.4f' % (df.OA_strict.std(), df.mIoU_strict.std()))
print()
t = pd.DataFrame({'support_px': sup.astype(int), 'precision': prec, 'recall': rec, 'IoU': iou, 'F1': f1}, index=NAMES).round(3)
print('Pooled over 12 months, per class:')
print(t.to_string())
print()
print('Green (unspecified) pixels predicted as trees or low_veg: %.1f %%' % (100 * gok_t / max(gall_t, 1)))
ref_share = 100 * sup / sup.sum()
pred_share = 100 * pooled[:, :5].sum(0) / pooled.sum()
print()
print('Class share on scored pixels (reference vs model, %):')
print(pd.DataFrame({'reference': ref_share, 'model': pred_share, 'diff_pp': pred_share - ref_share}, index=NAMES).round(2).to_string())

cm = pooled / np.maximum(pooled.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(6)); ax.set_xticklabels(PRED); ax.set_yticks(range(5)); ax.set_yticklabels(NAMES)
for i in range(5):
    for j in range(6):
        ax.text(j, i, f'{100 * cm[i, j]:.0f}%', ha='center', va='center',
                color='white' if cm[i, j] > 0.5 else 'black')
ax.set_xlabel('Model prediction'); ax.set_ylabel('Reference (URA + OSM)')
ax.set_title('Singapore 2020: confusion matrix (row = % of reference class)')
fig.colorbar(im); fig.tight_layout()
fig.savefig('outputs/singapore_2020-01/confusion_vs_reference.png', dpi=110)
print('saved outputs/singapore_2020-01/evaluation_per_month.csv and confusion_vs_reference.png')
