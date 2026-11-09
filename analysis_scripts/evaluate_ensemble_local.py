"""Ensemble and calibration evaluation on the TEST set, on CPU (same scoring as the Kaggle notebooks).
Needs X_test.npy, Y_test.npy, meta_test.npz in one folder (downloaded from the Kaggle notebook output).
Usage:  python evaluate_ensemble_local.py --limit 64          (quick test, about 1 minute)
        python evaluate_ensemble_local.py                     (full test set, about 20-40 minutes)
Scoring as in the Kaggle evaluation: label = consensus layer Y[:,0], values >= 9 ignored, mIoU over classes
present in the reference, bootstrap over places (2000 resamples, seed 0)."""
import argparse, os, time
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lulc_model import load_model, load_norm, MODELS, CLASSES, weights_path

FINE = 300                      # fine certainty bins (15 reporting bins = 20 fine bins each)
NB = 15


def metrics(c):
    d, row, col = np.diag(c), c.sum(1), c.sum(0)
    iou = d / np.maximum(row + col - d, 1)
    f1 = 2 * d / np.maximum(row + col, 1)
    return d.sum() / max(c.sum(), 1), iou[row > 0].mean(), iou, f1, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'C:\Users\Vinay\Desktop\lulc-eval')
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=0, help='only the first N valid patches (quick test)')
    ap.add_argument('--batch', type=int, default=16)
    a = ap.parse_args()
    out = a.out or os.path.join(a.data, 'results_ensemble' + ('_test' if a.limit else ''))
    os.makedirs(out, exist_ok=True)

    names = [n for n in MODELS if os.path.exists(weights_path(n))]
    assert len(names) == 5, f'need all 5 weights in weights/, found {names}'
    models = {n: load_model(n) for n in names}
    mean, std = load_norm()
    Xm = np.load(os.path.join(a.data, 'X_test.npy'), mmap_mode='r')
    Ym = np.load(os.path.join(a.data, 'Y_test.npy'), mmap_mode='r')
    meta = np.load(os.path.join(a.data, 'meta_test.npz'))
    vi = np.where(meta['valid'])[0]
    if a.limit:
        vi = vi[:a.limit]
    uniq = np.unique(meta['pid'][vi])
    loc = {p: i for i, p in enumerate(uniq.tolist())}
    pid_loc = np.array([loc[p] for p in meta['pid'][vi].tolist()])
    nP = len(uniq)
    print(f'valid test patches: {len(vi)} | places: {nP} | models: {names}')

    keys = ['ensemble'] + names
    cp = {k: np.zeros(nP * 81, np.int64) for k in keys}
    cal = {k: dict(n=np.zeros(FINE), ok=np.zeros(FINE), conf=np.zeros(FINE)) for k in ('ensemble', 'unet_pre')}
    ag_n, ag_ok = np.zeros(6), np.zeros(6)
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(vi), a.batch):
            ii = vi[s:s + a.batch]
            x = (np.asarray(Xm[ii], dtype=np.float32) - mean) / std
            x = torch.from_numpy(np.nan_to_num(x, nan=0.0))
            yc = np.asarray(Ym[ii][:, 0]).astype(np.int64)
            mk = yc < 9
            pl = pid_loc[s:s + a.batch].reshape(-1, 1, 1) * np.ones_like(yc)
            probs = {n: torch.softmax(models[n](x), 1) for n in names}
            ens = torch.stack(list(probs.values())).mean(0)
            pred = {n: probs[n].argmax(1).numpy() for n in names}
            conf = {n: probs[n].max(1).values.numpy() for n in names}
            pred['ensemble'], conf['ensemble'] = ens.argmax(1).numpy(), ens.max(1).values.numpy()
            for k in keys:
                cp[k] += np.bincount((pl * 81 + yc * 9 + pred[k])[mk], minlength=nP * 81)
            for k in cal:
                bi = np.minimum((conf[k] * FINE).astype(int), FINE - 1)[mk]
                ok = (pred[k] == yc)[mk]
                cal[k]['n'] += np.bincount(bi, minlength=FINE)
                cal[k]['ok'] += np.bincount(bi, weights=ok, minlength=FINE)
                cal[k]['conf'] += np.bincount(bi, weights=conf[k][mk], minlength=FINE)
            agree = sum((pred[n] == pred['ensemble']).astype(int) for n in names)
            ok = (pred['ensemble'] == yc)[mk]
            ag_n += np.bincount(agree[mk], minlength=6)
            ag_ok += np.bincount(agree[mk], weights=ok, minlength=6)
            done = s + len(ii)
            if (s // a.batch) % 10 == 0 or done == len(vi):
                el = time.time() - t0
                print(f'{done}/{len(vi)} patches | {el / 60:.1f} min | ETA {el / done * (len(vi) - done) / 60:.1f} min', flush=True)

    Cp = {k: cp[k].reshape(nP, 9, 9).astype(np.float64) for k in keys}
    # ---- accuracy and bootstrap
    rng = np.random.default_rng(0)
    boot = {k: [] for k in keys}
    for _ in range(2000):
        idx = rng.integers(0, nP, nP)
        for k in keys:
            boot[k].append(metrics(Cp[k][idx].sum(0))[1])
    rows = []
    for k in keys:
        oa, mi, _, _, _ = metrics(Cp[k].sum(0))
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        rows.append(dict(model=k, OA=round(oa, 3), mIoU=round(mi, 3), mIoU_lo=round(lo, 3), mIoU_hi=round(hi, 3)))
    comp = pd.DataFrame(rows)
    d = np.array(boot['ensemble']) - np.array(boot['unet_pre'])
    diff = metrics(Cp['ensemble'].sum(0))[1] - metrics(Cp['unet_pre'].sum(0))[1]
    comp.to_csv(os.path.join(out, 'model_comparison_local.csv'), index=False)
    print('\n', comp.to_string(index=False))
    print(f'\nensemble minus unet_pre mIoU: {diff:+.3f}  95% CI [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]')
    pc = pd.DataFrame(index=CLASSES)
    for k in ('unet_pre', 'ensemble'):
        _, _, iou, f1, row = metrics(Cp[k].sum(0))
        pc[f'{k}_IoU'], pc[f'{k}_F1'] = iou.round(3), f1.round(3)
    pc['reference_pixels'] = metrics(Cp['unet_pre'].sum(0))[4].astype(int)
    pc.to_csv(os.path.join(out, 'per_class_unet_vs_ensemble.csv'))
    print('\n', pc.to_string())
    # ---- calibration
    summ, curves = [], {}
    for k, v in cal.items():
        n15 = v['n'].reshape(NB, -1).sum(1)
        ok15 = v['ok'].reshape(NB, -1).sum(1)
        cf15 = v['conf'].reshape(NB, -1).sum(1)
        acc = np.where(n15 > 0, ok15 / np.maximum(n15, 1), np.nan)
        mc = np.where(n15 > 0, cf15 / np.maximum(n15, 1), np.nan)
        ece = np.nansum(np.abs(acc - mc) * n15 / n15.sum())
        curves[k] = (mc, acc, n15)
        summ.append(dict(model=k, ECE_pp=round(100 * ece, 2), mean_certainty_pct=round(100 * v['conf'].sum() / v['n'].sum(), 1),
                         accuracy_pct=round(100 * v['ok'].sum() / v['n'].sum(), 1)))
        pd.DataFrame(dict(bin_lo=np.arange(NB) / NB, bin_hi=(np.arange(NB) + 1) / NB, pixels=n15.astype(int),
                          mean_certainty=mc.round(4), accuracy=acc.round(4))).to_csv(
            os.path.join(out, f'calibration_bins_{k}.csv'), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(out, 'calibration_summary.csv'), index=False)
    print('\n', pd.DataFrame(summ).to_string(index=False))
    v = cal['ensemble']
    th = []
    for t in (0.5, 0.7, 0.8, 0.9, 0.93, 0.95, 0.99):
        j = int(round(t * FINE))
        nn, kk = v['n'][j:].sum(), v['ok'][j:].sum()
        th.append(dict(certainty_at_least=t, share_of_pixels_pct=round(100 * nn / v['n'].sum(), 1), accuracy_pct=round(100 * kk / max(nn, 1), 1)))
    pd.DataFrame(th).to_csv(os.path.join(out, 'ensemble_accuracy_by_certainty.csv'), index=False)
    print('\n', pd.DataFrame(th).to_string(index=False))
    ag = pd.DataFrame(dict(models_agreeing_with_ensemble=range(6), pixels_pct=(100 * ag_n / ag_n.sum()).round(1),
                           ensemble_accuracy_pct=(100 * ag_ok / np.maximum(ag_n, 1)).round(1)))
    ag.to_csv(os.path.join(out, 'ensemble_agreement.csv'), index=False)
    print('\n', ag.to_string(index=False))
    # ---- figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot([0, 1], [0, 1], '--', color='gray', label='perfect')
    for k, c in (('unet_pre', '#d9534f'), ('ensemble', '#2b6cb0')):
        mc, acc, n15 = curves[k]
        ax[0].plot(mc, acc, 'o-', color=c, label=k)
    ax[0].set_xlabel('stated certainty'); ax[0].set_ylabel('actual accuracy'); ax[0].set_title('Reliability (test set)')
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].bar(ag.models_agreeing_with_ensemble, ag.ensemble_accuracy_pct, color='#2b6cb0')
    for i, (p, q) in enumerate(zip(ag.pixels_pct, ag.ensemble_accuracy_pct)):
        ax[1].text(i, q + 1, f'{p}% of px', ha='center', fontsize=8)
    ax[1].set_xlabel('number of the 5 models that agree with the ensemble'); ax[1].set_ylabel('ensemble accuracy (%)')
    ax[1].set_title('Agreement of the 5 models'); ax[1].set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'reliability_and_agreement.png'), dpi=130)
    print('\nsaved in', out, f'| total {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
