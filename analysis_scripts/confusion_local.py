"""Confusion matrix of the final U-Net (unet_pre) on the TEST set, on CPU (about 3-5 minutes).
Same scoring as evaluate_ensemble_local.py. Saves confusion_unet_pre.csv (rows = reference, columns = prediction)."""
import argparse, os, time
import numpy as np, pandas as pd, torch
from lulc_model import load_model, load_norm, CLASSES

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=r'C:\Users\Vinay\Desktop\lulc-eval')
ap.add_argument('--model', default='unet_pre')
a = ap.parse_args()
model, (mean, std) = load_model(a.model), load_norm()
Xm = np.load(os.path.join(a.data, 'X_test.npy'), mmap_mode='r')
Ym = np.load(os.path.join(a.data, 'Y_test.npy'), mmap_mode='r')
vi = np.where(np.load(os.path.join(a.data, 'meta_test.npz'))['valid'])[0]
c = np.zeros(81, np.int64)
t0 = time.time()
with torch.inference_mode():
    for s in range(0, len(vi), 16):
        ii = vi[s:s + 16]
        x = torch.from_numpy(np.nan_to_num((np.asarray(Xm[ii], dtype=np.float32) - mean) / std, nan=0.0))
        y = np.asarray(Ym[ii][:, 0]).astype(np.int64)
        p = model(x).argmax(1).numpy()
        mk = y < 9
        c += np.bincount((y * 9 + p)[mk], minlength=81)
        if (s // 16) % 40 == 0:
            print(f'{s + len(ii)}/{len(vi)} | {(time.time() - t0) / 60:.1f} min', flush=True)
df = pd.DataFrame(c.reshape(9, 9), index=CLASSES, columns=CLASSES)
out = os.path.join(a.data, f'confusion_{a.model}.csv')
df.to_csv(out)
print('saved', out)
print('row-normalised (% of reference pixels):')
print((100 * df.div(df.sum(axis=1).clip(lower=1), axis=0)).round(1).to_string())
