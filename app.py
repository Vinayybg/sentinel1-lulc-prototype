import os, datetime
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go

from fetch_s1 import fetch_series, validate_inputs, LulcError
from lulc_model import (load_model, predict_series, available_models, MODELS, DEFAULT_MODEL, ENSEMBLE,
                        reliability_table, CLASSES, NODATA)
from lulc_stats import trend_table
from lulc_report import report, month_labels, COLORS, NODATA_COLOR
from lulc_smooth import smooth_maps, persistent_tables

st.set_page_config(page_title='Sentinel-1 LULC Monitor', page_icon='🛰️', layout='wide')

PRESETS = {
    'Singapore (urban, case study)': (1.3521, 103.8198),
    'Auckland (coastal city)': (-36.8485, 174.7633),
    'Custom location': (1.3521, 103.8198),
}
ALL_NAMES = CLASSES + ['no data']
ALL_COLORS = COLORS + [NODATA_COLOR]
FONT = dict(family='Inter, Segoe UI, sans-serif', size=13)
TRANSPARENT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=FONT)

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px;}
h2, h3 {letter-spacing: -0.01em;}
.hero {background: linear-gradient(120deg, #0b2e4a 0%, #14639c 55%, #22a094 100%);
       border-radius: 18px; padding: 28px 34px 24px 34px; color: #fff; margin-bottom: 20px;
       box-shadow: 0 8px 28px rgba(0,0,0,0.30);}
.hero h1 {margin: 0; font-size: 2.0rem; font-weight: 750; color: #fff; letter-spacing: -0.02em;}
.hero p {margin: 8px 0 14px 0; font-size: 1.0rem; opacity: 0.93; color: #fff; max-width: 900px;}
.chip {display: inline-block; padding: 4px 12px; margin: 0 8px 4px 0; border-radius: 999px;
       background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28);
       font-size: 0.78rem; color: #fff; letter-spacing: .02em;}
.kpi {border: 1px solid rgba(160,174,192,0.20); background: #14212e; border-radius: 14px;
      padding: 16px 18px; height: 100%; border-left: 5px solid var(--accent, #2a9d8f);
      box-shadow: 0 2px 10px rgba(0,0,0,0.20);}
.kpi .lab {font-size: 0.74rem; text-transform: uppercase; letter-spacing: .08em; opacity: .65;}
.kpi .val {font-size: 1.75rem; font-weight: 750; margin-top: 3px; line-height: 1.15;}
.kpi .sub {font-size: 0.82rem; opacity: .68; margin-top: 3px;}
.legend span {display: inline-block; margin: 2px 14px 2px 0; font-size: 0.85rem; opacity: .9;}
.legend i {display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px;}
.note {border-left: 4px solid #2a9d8f; background: rgba(42,157,143,0.12);
       padding: 12px 16px; border-radius: 8px; font-size: 0.92rem; margin-top: 10px;}
.section {font-size: 1.05rem; font-weight: 650; margin: 6px 0 2px 0;}
.foot {margin-top: 34px; padding-top: 14px; border-top: 1px solid rgba(160,174,192,0.20);
       font-size: 0.8rem; opacity: .65; line-height: 1.6;}
section[data-testid="stSidebar"] {border-right: 1px solid rgba(160,174,192,0.15);}
button[kind="primary"] {font-weight: 650; letter-spacing: .02em;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <h1>🛰️ Sentinel-1 Land Use / Land Cover Monitor</h1>
  <p>Cloud-independent radar monitoring. Enter any place on Earth and a start month to get
  12 monthly land-cover maps, class shares and change in percentage points, from Sentinel-1 radar only.</p>
  <span class="chip">Sentinel-1 SAR · VV + VH</span><span class="chip">9 land-cover classes</span>
  <span class="chip">U-Net · DeepLabV3+ · FPN</span><span class="chip">Global coverage</span>
  <span class="chip">12 monthly maps</span>
</div>
""", unsafe_allow_html=True)


def hex2rgb(h):
    h = h.lstrip('#')
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def kpi(col, label, value, sub='', accent='#2a9d8f'):
    col.markdown(f'<div class="kpi" style="--accent:{accent}"><div class="lab">{label}</div>'
                 f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


def legend_html():
    items = ''.join(f'<span><i style="background:{c}"></i>{n}</span>' for c, n in zip(ALL_COLORS, ALL_NAMES))
    return f'<div class="legend">{items}</div>'


def map_figure(m, title, step=2, res_m=10):
    z = np.where(m == NODATA, 9, m)[::step, ::step]
    km = np.arange(z.shape[0]) * res_m * step / 1000.0
    scale = []
    for i, c in enumerate(ALL_COLORS):
        scale += [[i / 10, c], [(i + 1) / 10, c]]
    names = np.array(ALL_NAMES)[z]
    fig = go.Figure(go.Heatmap(z=z, x=km, y=km, zmin=-0.5, zmax=9.5, colorscale=scale, showscale=False,
                               text=names, hovertemplate='%{text}<br>east %{x:.2f} km · south %{y:.2f} km<extra></extra>'))
    fig.update_layout(title=title, height=640, margin=dict(l=10, r=10, t=46, b=10),
                      xaxis=dict(title='km (west → east)', showgrid=False, constrain='domain'),
                      yaxis=dict(title='km (north → south)', autorange='reversed', scaleanchor='x',
                                 showgrid=False), **TRANSPARENT)
    return fig


@st.cache_resource
def get_model(name):
    return load_model(name)


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown('### 📍 Location')
    if 'preset' not in st.session_state:
        st.session_state['preset'] = list(PRESETS.keys())[0]
        st.session_state['lat'], st.session_state['lon'] = PRESETS[st.session_state['preset']]

    def _apply_preset():
        if st.session_state['preset'] != 'Custom location':
            st.session_state['lat'], st.session_state['lon'] = PRESETS[st.session_state['preset']]

    def _to_custom():
        st.session_state['preset'] = 'Custom location'

    preset = st.selectbox('Preset', list(PRESETS.keys()), key='preset', on_change=_apply_preset)
    lat = st.number_input('Latitude', -90.0, 90.0, format='%.4f', key='lat', on_change=_to_custom)
    lon = st.number_input('Longitude', -180.0, 180.0, format='%.4f', key='lon', on_change=_to_custom)
    st.markdown('### 🗓️ Period')
    today = datetime.date.today()
    year = st.number_input('Start year', 2015, today.year, 2020, step=1)
    month = st.selectbox('Start month', list(range(1, 13)), index=0,
                         format_func=lambda m: datetime.date(2000, m, 1).strftime('%B'))
    st.markdown('### 🧠 Model')
    names = available_models()
    if not names:
        st.error('No weights found in the weights/ folder. See README, section "Weights".')
        st.stop()
    model_name = st.selectbox('Model', names, index=names.index(DEFAULT_MODEL) if DEFAULT_MODEL in names else 0,
                              format_func=lambda n: ('ensemble  (test mIoU 0.653, about 5x slower)' if n == ENSEMBLE
                                                     else f"{n}  (test mIoU {MODELS[n]['test_miou']:.3f})"),
                              help='unet_pre is the final model used in the report. The others are the compared variants; '
                                   'the ensemble averages all of them and is about 5 times slower.')
    st.caption('Average of the class probabilities of all available models.' if model_name == ENSEMBLE
               else MODELS[model_name]['label'])
    smooth = st.checkbox('Temporal smoothing (3-month majority vote)', value=False, help='Halves the month-to-month flicker; changes shorter than 2 months are suppressed.')
    run = st.button('▶  Run analysis', type='primary', width='stretch')
    st.caption('Analysed area: 10.24 km × 10.24 km around the point, 10 m pixels.')

# ---------------- compute ----------------
if run:
    try:
        lat, lon, year, month = validate_inputs(lat, lon, year, month)
    except LulcError as e:
        st.error(str(e))
        st.stop()
    lines, box = [], st.empty()

    def log(s):
        lines.append(s)
        box.code('\n'.join(lines))

    try:
        with st.spinner('Fetching 14 monthly Sentinel-1 composites from Earth Engine (cached after the first run)...'):
            series = fetch_series(lat, lon, int(year), int(month), log=log)
        with st.spinner('Running the model...'):
            maps, conf = predict_series(get_model(model_name), series, log=lambda s: None, return_conf=True)
    except LulcError as e:
        st.error(str(e))
        st.stop()
    if smooth:
        maps = smooth_maps(maps)
    box.empty()
    outdir = f'outputs/{lat:.4f}_{lon:.4f}_{int(year)}-{int(month):02d}_{model_name}' + ('_smoothed' if smooth else '')
    pct, step, fl = report(maps, int(year), int(month), outdir, f'LULC {lat:.4f}, {lon:.4f} ({model_name})')
    st.session_state['res'] = dict(maps=maps, conf=conf, pct=pct, step=step, fl=fl, lat=lat, lon=lon,
                                   year=int(year), month=int(month),
                                   arch=model_name + (' + 3-month smoothing' if smooth else ''), outdir=outdir,
                                   labels=month_labels(int(year), int(month)))

res = st.session_state.get('res')
if res is None:
    st.info('Choose a location and period in the sidebar, then press **Run analysis**.')
    st.stop()

maps, pct, step, fl, labels = res['maps'], res['pct'], res['step'], res['fl'], res['labels']
conf = res.get('conf')
P = pct[CLASSES]
mean_share = P.mean()
dom = mean_share.idxmax()
valid_months = int((pct['valid_%'] > 0).sum())
big = fl.iloc[0].abs().idxmax()

st.markdown(f"**{res['lat']:.4f}, {res['lon']:.4f}**  ·  {labels[0]} to {labels[-1]}  ·  model: {res['arch']}")
c = st.columns(5)
kpi(c[0], 'Dominant class', dom.capitalize(), f'{mean_share[dom]:.1f}% of the area on average',
    COLORS[CLASSES.index(dom)])
kpi(c[1], 'Built-up', f"{mean_share['built']:.1f}%", 'average over 12 months', COLORS[6])
kpi(c[2], 'Tree cover', f"{mean_share['trees']:.1f}%", 'average over 12 months', COLORS[1])
kpi(c[3], 'Largest net change', f"{fl.iloc[0][big]:+.2f} pp", f'{big}, {labels[0]} → {labels[-1]}', '#ed8936')
kpi(c[4], 'Months with data', f'{valid_months} / 12', f"valid pixels ≥ {pct['valid_%'].min():.0f}%", '#4299e1')
st.write('')
bad = pct.index[pct['valid_%'] < 100].tolist()
if bad:
    st.warning('Months with missing data (shown as "no data", never as "no change"): '
               + ', '.join(f'{m} ({pct.loc[m, "valid_%"]}% valid)' for m in bad))

tab1, tab2, tab3, tab4, tab5 = st.tabs(['🗺️ Overview', '📈 Trends', '🔁 Change', '📋 Data', 'ℹ️ Method & limits'])

# ---------------- overview ----------------
with tab1:
    left, right = st.columns([3, 2], gap='large')
    with left:
        sel = st.select_slider('Month', options=labels, value=labels[0])
        k = labels.index(sel)
        show_conf = conf is not None and st.toggle('Show model certainty instead of classes', value=False,
                                                   help='Probability the model gives to the class it chose. Dark = unsure.')
        if show_conf:
            cz = conf[k].astype(np.float32)
            km = np.arange(cz.shape[0]) * 10 * 2 / 1000.0
            figc = go.Figure(go.Heatmap(z=cz[::2, ::2], x=km, y=km, zmin=0.3, zmax=1.0, colorscale='Viridis',
                                        colorbar=dict(title='certainty'),
                                        hovertemplate='certainty %{z:.2f}<extra></extra>'))
            figc.update_layout(title=f'Model certainty, {sel}', height=640, margin=dict(l=10, r=10, t=46, b=10),
                               xaxis=dict(title='km (west → east)', showgrid=False, constrain='domain'),
                               yaxis=dict(title='km (north → south)', autorange='reversed', scaleanchor='x',
                                          showgrid=False), **TRANSPARENT)
            st.plotly_chart(figc, width='stretch')
        else:
            st.plotly_chart(map_figure(maps[k], f'Land cover, {sel}'), width='stretch')
            st.markdown(legend_html(), unsafe_allow_html=True)
        if conf is not None:
            cv = conf[k][np.isfinite(conf[k].astype(np.float32))].astype(np.float32)
            if cv.size:
                st.caption(f'Mean certainty {100 * cv.mean():.0f}% · {100 * (cv < 0.6).mean():.0f}% of pixels below 60% certainty')
                if not res['arch'].startswith(ENSEMBLE):
                    st.caption('A single model is over-confident: on the test set the final U-Net states 87% certainty on average '
                               'but is right 80% of the time. The ensemble is well calibrated (84% stated, 82% right).')
    with right:
        vals = P.iloc[k].values
        fig = go.Figure(go.Bar(x=vals, y=CLASSES, orientation='h', marker_color=COLORS,
                               text=[f'{v:.1f}%' for v in vals], textposition='outside', cliponaxis=False))
        fig.update_layout(title=f'Class share, {sel}', height=330, margin=dict(l=10, r=60, t=46, b=10),
                          yaxis=dict(autorange='reversed'),
                          xaxis=dict(title='% of valid pixels', range=[0, max(float(np.nanmax(vals)) * 1.3, 5)]),
                          **TRANSPARENT)
        st.plotly_chart(fig, width='stretch')
        a, b = maps[0], maps[-1]
        both = (a != NODATA) & (b != NODATA)
        chg = (a != b) & both
        d = np.full(a.shape + (3,), hex2rgb('#e2e8f0'), np.uint8)
        d[chg] = hex2rgb('#f6ad55')
        d[~both] = hex2rgb(NODATA_COLOR)
        st.image(d, caption=f'Pixels with a different class, {labels[0]} vs {labels[-1]} (orange)', width=360)
        st.markdown(f'<div class="note"><b>{100 * chg.sum() / max(both.sum(), 1):.1f}%</b> of pixels changed class '
                    f'between the first and last month, while the net class shares moved by only about '
                    f'<b>{fl.iloc[0].abs().max():.1f} pp</b>. Most of this is classification noise that cancels '
                    f'out in the net numbers.</div>', unsafe_allow_html=True)

# ---------------- trends ----------------
with tab2:
    fig = go.Figure()
    for cname, col in zip(CLASSES, COLORS):
        fig.add_trace(go.Scatter(x=labels, y=P[cname], mode='lines', name=cname, stackgroup='one',
                                 line=dict(width=0.6, color=col), fillcolor=col))
    fig.update_layout(title='Class composition over the 12 months', height=430, yaxis_title='% of valid pixels',
                      yaxis_range=[0, 100], margin=dict(l=10, r=10, t=50, b=10), **TRANSPARENT)
    st.plotly_chart(fig, width='stretch')
    default = [cn for cn in CLASSES if mean_share[cn] > 0.5]
    pick = st.multiselect('Classes to zoom in on', CLASSES, default=default)
    fig2 = go.Figure()
    for cname in pick:
        fig2.add_trace(go.Scatter(x=labels, y=P[cname], mode='lines+markers', name=cname,
                                  line=dict(color=COLORS[CLASSES.index(cname)], width=2.5)))
    fig2.update_layout(height=360, yaxis_title='% of valid pixels', margin=dict(l=10, r=10, t=20, b=10),
                       **TRANSPARENT)
    st.plotly_chart(fig2, width='stretch')

# ---------------- change ----------------
with tab3:
    l2, r2 = st.columns([3, 2], gap='large')
    with l2:
        z = step.T.values
        fig = go.Figure(go.Heatmap(z=z, x=[s.replace(' -> ', ' → ') for s in step.index], y=CLASSES,
                                   zmid=0, zmin=-3, zmax=3, colorscale='RdBu',
                                   text=np.round(z, 1), texttemplate='%{text}', colorbar=dict(title='pp')))
        fig.update_layout(title='Month-to-month change (percentage points)', height=430,
                          yaxis=dict(autorange='reversed'), margin=dict(l=10, r=10, t=50, b=10), **TRANSPARENT)
        st.plotly_chart(fig, width='stretch')
    with r2:
        v = fl.iloc[0].values
        fig = go.Figure(go.Bar(x=CLASSES, y=v, marker_color=COLORS,
                               text=[f'{x:+.2f}' for x in v], textposition='outside', cliponaxis=False))
        fig.add_hline(y=0, line_width=1)
        fig.update_layout(title=f'First to last month: {labels[0]} → {labels[-1]}', height=430,
                          yaxis_title='change in percentage points', margin=dict(l=10, r=10, t=50, b=10),
                          **TRANSPARENT)
        st.plotly_chart(fig, width='stretch')
    st.markdown('<div class="note">Single-month changes of a few tenths of a percentage point are classifier noise. '
                'The trend test below decides which 12-month changes are large and consistent enough to be worth checking.</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section">Is the change real? 12-month trend test per class</div>', unsafe_allow_html=True)
    tt = trend_table(P)
    st.dataframe(tt, width='stretch')
    sig = tt.index[tt['verdict'] == 'trend'].tolist()
    st.markdown('<div class="note">' + ('Classes with a detectable trend: <b>' + ', '.join(sig) + '</b>. ' if sig else
                '<b>No class shows a detectable trend</b>: all changes are within the monthly noise. ')
                + 'A change only counts if the Mann-Kendall test is significant (p &lt; 0.05) and the change is larger than twice '
                'the monthly noise (and at least 0.5 pp). Twelve months cannot separate a real trend from a seasonal swing, '
                'so treat a "trend" as "worth checking".</div>', unsafe_allow_html=True)
    st.write('')
    summ, pnet, ptrans = persistent_tables(maps, labels)
    st.markdown('<div class="section">Persistent change (one-month flicker ignored)</div>', unsafe_allow_html=True)
    s0 = summ.iloc[0]
    k1, k2, k3 = st.columns(3)
    kpi(k1, 'Raw: first vs last month', f"{s0['raw_first_vs_last_month_changed_%']:.1f}%",
        'pixels with a different class (mostly noise)', '#ed8936')
    kpi(k2, 'Persistent change', f"{s0['persistent_changed_%']:.1f}%",
        'first 3 months vs last 3 months, both decided', '#2a9d8f')
    kpi(k3, 'Net change', f"{s0['net_change_%_of_area']:.1f}%",
        'share of the area that moved between classes, net', '#4299e1')
    st.write('')
    p1, p2 = st.columns(2, gap='large')
    with p1:
        st.caption('Per class: share at the start and the end (3-month majority), net change in percentage points, and gross flows')
        st.dataframe(pnet, width='stretch')
    with p2:
        st.caption('Largest persistent transitions (% of the analysed area)')
        st.dataframe(ptrans, width='stretch', hide_index=True)

# ---------------- data ----------------
with tab4:
    st.markdown('<div class="section">Class percentages per month</div>', unsafe_allow_html=True)
    st.dataframe(pct, width='stretch')
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button('⬇ Percentages (CSV)', pct.to_csv().encode(), 'class_percentages.csv', width='stretch')
    d2.download_button('⬇ Month-to-month (CSV)', step.to_csv().encode(), 'change_month_to_month_pp.csv', width='stretch')
    d3.download_button('⬇ First-to-last (CSV)', fl.to_csv().encode(), 'change_first_to_last_pp.csv', width='stretch')
    png = os.path.join(res['outdir'], 'maps_12_months.png')
    if os.path.exists(png):
        d4.download_button('⬇ 12-month figure (PNG)', open(png, 'rb').read(), 'maps_12_months.png',
                           mime='image/png', width='stretch')
    st.markdown('<div class="section">Month-to-month change (percentage points)</div>', unsafe_allow_html=True)
    st.dataframe(step, width='stretch')
    st.markdown('<div class="section">First to last month (percentage points)</div>', unsafe_allow_html=True)
    st.dataframe(fl, width='stretch')
    st.markdown('<div class="section">Persistent change, first 3 vs last 3 months</div>', unsafe_allow_html=True)
    _s, _n, _t = persistent_tables(maps, labels)
    st.dataframe(_s, width='stretch', hide_index=True)
    st.dataframe(_n, width='stretch')
    st.dataframe(_t, width='stretch', hide_index=True)
    e1, e2, e3 = st.columns(3)
    e1.download_button('⬇ Persistent summary (CSV)', _s.to_csv(index=False).encode(), 'persistent_change_summary.csv', width='stretch')
    e2.download_button('⬇ Persistent per class (CSV)', _n.to_csv().encode(), 'persistent_change_per_class.csv', width='stretch')
    e3.download_button('⬇ Persistent transitions (CSV)', _t.to_csv(index=False).encode(), 'persistent_change_transitions.csv', width='stretch')
    st.caption(f"All results are also saved in the folder: {res['outdir']}")

# ---------------- method ----------------
with tab5:
    st.markdown("""
**How it works**
1. Sentinel-1 IW scenes (VV and VH, in dB, from Google Earth Engine) are combined into a monthly median for a 10.24 km square.
2. For each map month *t*, the model sees three composites (*t−1*, *t*, *t+1*), so 12 maps need 14 monthly composites.
3. A segmentation network (default: ResNet-18 U-Net with a pretrained encoder; DeepLabV3+ and FPN are selectable) classifies the whole area with overlapping 256 × 256 windows, blended so there are no tile seams, into 9 classes: water, trees, grass, flooded vegetation, crops, shrub, built-up, bare, snow.
4. Class shares are computed over valid pixels; changes are reported in percentage points (pp). A missing month is "no data", never "no change".
5. Persistent change counts a pixel as changed only if its class in the first 3 months differs from its class in the last 3 months (majority of at least 2 of 3 months), so one-month flicker is ignored.

**How far to trust each class** (final model, test set, 147 places): see the table below.

**Limits to keep in mind**
- Trained on a global sample of 1,124 places (2020–2021); unseen regions can perform worse.
- Radar confuses some classes (grass, crops and shrub; bare ground and built-up).
- Seasonal effects (rain, soil moisture, crop cycles) change backscatter without changing land cover.
- Small net changes are dominated by classification noise; see the Change tab.
""")

with tab5:
    st.dataframe(reliability_table(), width='stretch')
    st.caption('IoU and F1 of the final model per class on the held-out test places, measured against the consensus labels. '
               'Other models and other reference data give different numbers.')

st.markdown("""
<div class="foot">
Contains modified Copernicus Sentinel-1 data, processed with Google Earth Engine. Land-cover training labels derived from
ESA WorldCover, Esri 10 m Land Cover and Google Dynamic World. Deep Learning course project, SRH University Heidelberg.
</div>
""", unsafe_allow_html=True)