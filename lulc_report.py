import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from lulc_model import CLASSES, NODATA, class_percentages
from fetch_s1 import add_months

COLORS = ['#2b6cb0', '#1f6b2e', '#8bd35f', '#38b2ac', '#ecc94b',
          '#a3a33a', '#e53e3e', '#d2b48c', '#f7fafc']       # water..snow
NODATA_COLOR = '#2d2d2d'


def month_labels(y, m, n=12):
    return [f'{yy}-{mm:02d}' for yy, mm in (add_months(y, m, k) for k in range(n))]


def tables(maps, labels):
    P = np.array([class_percentages(m)[0] for m in maps])           # (n, 9)
    V = np.array([class_percentages(m)[1] for m in maps])           # valid %
    pct = pd.DataFrame(P, index=labels, columns=CLASSES).round(2)
    pct['valid_%'] = V.round(1)
    step = pd.DataFrame(np.diff(P, axis=0),
                        index=[f'{labels[i]} -> {labels[i + 1]}' for i in range(len(labels) - 1)],
                        columns=CLASSES).round(2)
    first_last = pd.DataFrame([P[-1] - P[0]], index=[f'{labels[0]} -> {labels[-1]}'],
                              columns=CLASSES).round(2)
    return pct, step, first_last


def plot_12(maps, labels, path, title=''):
    cmap = ListedColormap(COLORS + [NODATA_COLOR])
    n = len(maps)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows + 1))
    for ax in np.ravel(axes):
        ax.axis('off')
    for k in range(n):
        m = maps[k][::2, ::2].astype(int)                            # display only
        m = np.where(m == NODATA, 9, m)
        ax = np.ravel(axes)[k]
        ax.imshow(m, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        ax.set_title(labels[k], fontsize=11)
    handles = [Patch(color=c, label=n_) for c, n_ in zip(COLORS, CLASSES)]
    handles.append(Patch(color=NODATA_COLOR, label='no data'))
    fig.legend(handles=handles, loc='lower center', ncol=10, fontsize=9, frameon=False)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97), h_pad=2.5)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def report(maps, start_year, start_month, outdir, title=''):
    os.makedirs(outdir, exist_ok=True)
    labels = month_labels(start_year, start_month, len(maps))
    pct, step, fl = tables(maps, labels)
    pct.to_csv(os.path.join(outdir, 'class_percentages.csv'))
    step.to_csv(os.path.join(outdir, 'change_month_to_month_pp.csv'))
    fl.to_csv(os.path.join(outdir, 'change_first_to_last_pp.csv'))
    from lulc_stats import trend_table
    trend_table(pct[CLASSES]).to_csv(os.path.join(outdir, 'change_significance.csv'))
    if len(maps) >= 6:                                   # persistent change (one-month flicker ignored)
        from lulc_smooth import persistent_tables
        summ, net, trans = persistent_tables(maps, labels)
        summ.to_csv(os.path.join(outdir, 'persistent_change_summary.csv'), index=False)
        net.to_csv(os.path.join(outdir, 'persistent_change_per_class.csv'))
        trans.to_csv(os.path.join(outdir, 'persistent_change_transitions.csv'), index=False)
    plot_12(maps, labels, os.path.join(outdir, 'maps_12_months.png'), title)
    return pct, step, fl


if __name__ == '__main__':
    maps = np.load('cache/maps_singapore_2020.npy')
    pct, step, fl = report(maps, 2020, 1, 'outputs/singapore_2020-01',
                           'Singapore, 12 monthly LULC maps (Sentinel-1 only)')
    pd.set_option('display.width', 200)
    print(pct, '\n')
    print(step, '\n')
    print(fl)
    print('\nsaved to outputs/singapore_2020-01/')