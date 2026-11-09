"""Sentinel-1 LULC model loading and whole-area prediction.

Five trained models are available (see MODELS). The default is the final model,
the ImageNet-pretrained U-Net ResNet-18. Choose another one with the argument
`name`, the environment variable LULC_MODEL, or the --model option of run_lulc.py.
"""
import json, os, numpy as np, torch
import segmentation_models_pytorch as smp

BASE = os.path.dirname(os.path.abspath(__file__))
NORM = os.path.join(BASE, 'weights', 's1_norm.json')
CLASSES = ['water', 'trees', 'grass', 'flooded', 'crops', 'shrub', 'built', 'bare', 'snow']
NODATA = 255
TILE = 256

# name -> architecture, encoder, weights file, description, test mIoU vs consensus (147 test places)
MODELS = {
    'unet_pre': dict(arch='unet', enc='resnet18', file='E4_consensus_unet_pre_best.pt', test_miou=0.623,
                     label='U-Net ResNet-18, ImageNet-pretrained encoder (final model)'),
    'unet_scratch': dict(arch='unet', enc='resnet18', file='E3_consensus_unet_best.pt', test_miou=0.593,
                         label='U-Net ResNet-18, trained from scratch (no pretraining)'),
    'deeplab_pre': dict(arch='deeplab', enc='resnet18', file='E4_consensus_deeplab_pre_best.pt', test_miou=0.620,
                        label='DeepLabV3+ ResNet-18, ImageNet-pretrained encoder'),
    'deeplab_scratch': dict(arch='deeplab', enc='resnet18', file='E3_consensus_deeplab_best.pt', test_miou=0.609,
                            label='DeepLabV3+ ResNet-18, trained from scratch'),
    'fpn_r50_pre': dict(arch='fpn', enc='resnet50', file='E5_consensus_fpn_r50_best.pt', test_miou=0.630,
                        label='FPN ResNet-50, ImageNet-pretrained encoder'),
}
DEFAULT_MODEL = os.environ.get('LULC_MODEL', 'unet_pre')
ENSEMBLE = 'ensemble'          # average of the class probabilities of all available models

# Test-set reliability of the final model (unet_pre) per class, vs the consensus labels (147 test places):
# class -> (IoU, F1, note)
RELIABILITY = {
    'water':   (0.872, 0.932, 'reliable'),
    'trees':   (0.796, 0.886, 'reliable; sometimes mixed up with shrub'),
    'grass':   (0.466, 0.636, 'weak: often confused with bare ground, shrub and trees; usually underestimated'),
    'flooded': (0.524, 0.688, 'fair'),
    'crops':   (0.665, 0.799, 'fair'),
    'shrub':   (0.352, 0.521, 'weak: often confused with trees and grass'),
    'built':   (0.805, 0.892, 'reliable'),
    'bare':    (0.723, 0.839, 'reliable'),
    'snow':    (0.403, 0.575, 'weak, rare class'),
}


def reliability_table():
    import pandas as pd
    rows = []
    for c in CLASSES:
        iou, f1, note = RELIABILITY[c]
        rows.append((c, iou, f1, 'good' if iou >= 0.7 else 'fair' if iou >= 0.5 else 'weak', note))
    return pd.DataFrame(rows, columns=['class', 'test_IoU', 'test_F1', 'trust', 'note']).set_index('class')
NETS = {'unet': smp.Unet, 'deeplab': smp.DeepLabV3Plus, 'fpn': smp.FPN}


def weights_path(name):
    return os.path.join(BASE, 'weights', MODELS[name]['file'])


def available_models():
    """Names of the models whose weights file is present in weights/ (plus the ensemble if there are at least 2)."""
    names = [n for n in MODELS if os.path.exists(weights_path(n))]
    return names + ([ENSEMBLE] if len(names) >= 2 else [])


class _Ensemble(torch.nn.Module):
    """Averages the class probabilities of several models. forward() returns log-probabilities, so a
    softmax on top gives the averaged probabilities back."""
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        p = torch.stack([torch.softmax(m(x), 1) for m in self.models]).mean(0)
        return torch.log(p.clamp_min(1e-8))


def load_norm(path=NORM):
    d = json.load(open(path))
    mean = np.array(d['mean'], np.float32).ravel()
    std = np.array(d['std'], np.float32).ravel()
    if mean.size == 2:                      # VV, VH -> repeat for 3 months
        mean, std = np.tile(mean, 3), np.tile(std, 3)
    assert mean.size == 6 and std.size == 6, f'unexpected norm file: {d}'
    return mean.reshape(6, 1, 1), std.reshape(6, 1, 1)


def load_model(name=None, arch=None):
    """load_model('unet_pre') -> model in eval mode.
    Old style load_model(weights_path, arch) still works for custom weights."""
    name = name or DEFAULT_MODEL
    if name == ENSEMBLE:
        members = [n for n in MODELS if os.path.exists(weights_path(n))]
        if len(members) < 2:
            raise FileNotFoundError('The ensemble needs at least 2 weights files in weights/.')
        return _Ensemble([load_model(n) for n in members]).eval()
    if name in MODELS:
        spec = MODELS[name]
        arch, enc, path = spec['arch'], spec['enc'], weights_path(name)
    else:
        arch = arch or 'unet'
        enc = 'resnet50' if arch == 'fpn' else 'resnet18'
        path = name
    if not os.path.exists(path):
        raise FileNotFoundError(f'Weights file not found: {path}. See README, section "Weights".')
    model = NETS[arch](enc, encoder_weights=None, in_channels=6, classes=9)
    ck = torch.load(path, map_location='cpu')
    sd = ck
    if isinstance(ck, dict) and not all(torch.is_tensor(v) for v in ck.values()):
        for k in ('model', 'state_dict', 'model_state', 'model_state_dict'):
            if k in ck:
                sd = ck[k]
                break
        else:
            raise SystemExit(f'Unknown checkpoint layout, keys: {list(ck.keys())}')
    model.load_state_dict(sd)
    return model.eval()


def prepare(stack6, mean, std):
    """stack6: (6,H,W) dB, NaN = no data. Returns normalised float32 and valid mask."""
    valid = np.isfinite(stack6).all(axis=0)
    x = np.where(np.isfinite(stack6), stack6, -35.0)
    x = np.clip(x, -35, 5)
    return ((x - mean) / std).astype(np.float32), valid


def _starts(n, win, stride):
    s = list(range(0, max(n - win, 0) + 1, stride))
    if s[-1] != n - win and n > win:
        s.append(n - win)
    return s


def _window_weight(win=TILE):
    """Smooth weight: 1 in the middle of a window, small at its edges."""
    r = np.sin(np.pi * (np.arange(win) + 0.5) / win).astype(np.float32)
    return np.maximum(np.outer(r, r), 0.02)


@torch.no_grad()
def predict_stack(model, stack6, mean, std, batch=8, stride=TILE // 2, return_conf=False):
    """One month: (6,H,W) -> (H,W) uint8 class map, 255 where any input is missing.

    The whole area is predicted with overlapping 256x256 windows (stride 128 by
    default). Class probabilities of overlapping windows are blended with a smooth
    weight, so there are no visible tile seams. stride=256 reproduces the old
    non-overlapping tiling. With return_conf=True a second array is returned: the
    probability of the predicted class (0.11 .. 1) as a certainty measure."""
    x, valid = prepare(stack6, mean, std)
    _, H, W = x.shape
    pad_h, pad_w = max(TILE - H, 0), max(TILE - W, 0)
    if pad_h or pad_w:
        x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    _, Hp, Wp = x.shape
    ys, xs = _starts(Hp, TILE, stride), _starts(Wp, TILE, stride)
    pos = [(i, j) for i in ys for j in xs]
    w2 = _window_weight()
    acc = np.zeros((9, Hp, Wp), np.float32)
    wsum = np.zeros((Hp, Wp), np.float32)
    for b in range(0, len(pos), batch):
        chunk = pos[b:b + batch]
        t = torch.from_numpy(np.stack([x[:, i:i + TILE, j:j + TILE] for i, j in chunk]))
        p = torch.softmax(model(t), 1).numpy()
        for k, (i, j) in enumerate(chunk):
            acc[:, i:i + TILE, j:j + TILE] += p[k] * w2
            wsum[i:i + TILE, j:j + TILE] += w2
    prob = acc / wsum
    out = prob.argmax(0).astype(np.uint8)[:H, :W]
    out[~valid] = NODATA
    if not return_conf:
        return out
    conf = prob.max(0)[:H, :W].astype(np.float16)
    conf[~valid] = np.nan
    return out, conf


def predict_series(model, series, log=print, stride=TILE // 2, return_conf=False):
    """series: (n+2, 2, H, W) -> maps (n, H, W). Map k uses composites k-1, k, k+1.
    With return_conf=True returns (maps, confidence) where confidence is float16 (n, H, W)."""
    mean, std = load_norm()
    maps, confs = [], []
    for k in range(1, series.shape[0] - 1):
        s6 = np.concatenate([series[k - 1], series[k], series[k + 1]], axis=0)
        if return_conf:
            m, c = predict_stack(model, s6, mean, std, stride=stride, return_conf=True)
            maps.append(m); confs.append(c)
        else:
            maps.append(predict_stack(model, s6, mean, std, stride=stride))
        log(f'map {k}/{series.shape[0] - 2} done')
    return (np.stack(maps), np.stack(confs)) if return_conf else np.stack(maps)


def class_percentages(m):
    valid = m != NODATA
    n = valid.sum()
    if n == 0:
        return np.full(9, np.nan), 0.0
    return np.array([100 * (m == c).sum() / n for c in range(9)]), 100 * valid.mean()
