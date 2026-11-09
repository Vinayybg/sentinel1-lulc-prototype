"""Sentinel-1 monthly composites from Google Earth Engine, with a local cache.

Earth Engine is only contacted for months that are not already in cache/.
Setup (see README): earthengine authenticate, then set EE_PROJECT=<your project id>."""
import datetime, os, time
import numpy as np, pyproj

BASE = os.path.dirname(os.path.abspath(__file__))
SIZE, RES = 1024, 10          # 1024 x 1024 px at 10 m = 10.24 km square (4 x 4 patches of 256)
CACHE = os.path.join(BASE, 'cache')
S1_FIRST = (2014, 10)         # first month with Sentinel-1 GRD data in Earth Engine
MIN_MONTHS_WITH_DATA = 10     # of 14 composites; below this the maps are unreliable (warning only)


class LulcError(RuntimeError):
    """Problem with the input or the Earth Engine connection. The message is meant for the user."""


def add_months(y, m, k):
    t = y * 12 + (m - 1) + k
    return t // 12, t % 12 + 1


def month_list(start_year, start_month, n_maps=12):
    """Composites needed: one month before the first map to one month after the last (n_maps + 2)."""
    return [add_months(start_year, start_month, k) for k in range(-1, n_maps + 1)]


def validate_inputs(lat, lon, start_year, start_month, n_maps=12, today=None):
    """Raises LulcError with a clear sentence if the request cannot be served."""
    today = today or datetime.date.today()
    try:
        lat, lon, start_year, start_month = float(lat), float(lon), int(start_year), int(start_month)
    except (TypeError, ValueError):
        raise LulcError('Latitude, longitude, start year and start month must be numbers.')
    if not (-90 <= lat <= 90):
        raise LulcError(f'Latitude {lat} is outside -90 .. 90.')
    if not (-180 <= lon <= 180):
        raise LulcError(f'Longitude {lon} is outside -180 .. 180.')
    if not (1 <= start_month <= 12):
        raise LulcError(f'Start month {start_month} is not between 1 and 12.')
    first = month_list(start_year, start_month, n_maps)[0]
    last = month_list(start_year, start_month, n_maps)[-1]
    if first < S1_FIRST:
        raise LulcError(f'Sentinel-1 data start in {S1_FIRST[0]}-{S1_FIRST[1]:02d}. The analysis needs the month before '
                        f'the first map, so the earliest start month is 2014-11.')
    if last[0] * 12 + last[1] >= today.year * 12 + today.month:
        e = add_months(today.year, today.month, -(n_maps + 1))
        raise LulcError(f'The {n_maps} maps plus one month before and after must lie in completed months in the past. '
                        f'Latest possible start month: {e[0]}-{e[1]:02d}.')
    if abs(lat) > 84:
        raise LulcError('Sentinel-1 does not cover latitudes beyond about 84 degrees. Choose a location further from the pole.')
    return lat, lon, start_year, start_month


def grid_for(lat, lon, size=SIZE, res=RES):
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    x, y = pyproj.Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    half = size * res // 2
    return epsg, round(x / res) * res - half, round(y / res) * res + half


def cache_path(lat, lon, y, m):
    return os.path.join(CACHE, f's1_{lat:.4f}_{lon:.4f}_{y}-{m:02d}_{SIZE}.npy')


_EE_READY = False


def _init_ee():
    """Initialise Earth Engine once, with readable errors."""
    global _EE_READY
    if _EE_READY:
        return
    import ee
    project = os.environ.get('EE_PROJECT') or os.environ.get('GOOGLE_CLOUD_PROJECT')
    if not project:                       # written by setup.bat / setup.sh
        pf = os.path.join(BASE, 'ee_project.txt')
        if os.path.exists(pf):
            project = open(pf, encoding='utf-8').read().strip()
    if not project:
        raise LulcError('No Earth Engine project is set. Run setup.bat (Windows) or setup.sh (macOS/Linux) once, or set the '
                        'environment variable EE_PROJECT to your Google Cloud project id. '
                        'See README, section "Earth Engine setup".')
    try:
        ee.Initialize(project=project)
    except Exception as e:
        msg = str(e)
        if any(k in msg.lower() for k in ('credentials', 'authenticate', 'authorized', 'token', 'oauth')):
            raise LulcError('Earth Engine login is missing or expired. Run "earthengine authenticate" once in a terminal, '
                            'then start the analysis again.')
        if any(k in msg.lower() for k in ('permission', 'not registered', 'not been used', 'disabled', '403')):
            raise LulcError(f'Project "{project}" cannot use Earth Engine. Check that it is registered for Earth Engine '
                            f'(non-commercial use) and that the Earth Engine API is enabled. Details: {msg[:200]}')
        raise LulcError(f'Could not start Earth Engine: {msg[:300]}')
    _EE_READY = True


def _translate(e):
    msg = str(e)
    low = msg.lower()
    if any(k in low for k in ('quota', 'rate limit', 'too many', 'resource_exhausted', 'compute units', '429')):
        return LulcError('Earth Engine quota or rate limit reached. Wait a few minutes, or if the monthly quota is used up, '
                         'use another Earth Engine account or wait for the reset (see README, "Quota").')
    if any(k in low for k in ('timed out', 'timeout', 'connection', 'network', 'ssl', 'name resolution')):
        return LulcError('Network problem while contacting Earth Engine. Check the internet connection and run again '
                         '(months already downloaded are kept in cache/).')
    return LulcError(f'Earth Engine error: {msg[:300]}')


def fetch_month(lat, lon, y, m):
    """Returns (array (2,SIZE,SIZE) float32 in dB, n_scenes). No scenes -> all NaN."""
    p = cache_path(lat, lon, y, m)
    if os.path.exists(p):
        a = np.load(p)
        return a, -1                      # -1 = read from cache (scene count not stored)
    _init_ee()
    import ee
    try:
        epsg, x0, y0 = grid_for(lat, lon)
        start = ee.Date.fromYMD(y, m, 1)
        col = (ee.ImageCollection('COPERNICUS/S1_GRD')
               .filterDate(start, start.advance(1, 'month'))
               .filterBounds(ee.Geometry.Point([lon, lat]))
               .filter(ee.Filter.eq('instrumentMode', 'IW'))
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
               .select(['VV', 'VH']))
        n = col.size().getInfo()
        if n == 0:
            a = np.full((2, SIZE, SIZE), np.nan, np.float32)
        else:
            r = ee.data.computePixels({
                'expression': col.median(), 'fileFormat': 'NUMPY_NDARRAY',
                'grid': {'dimensions': {'width': SIZE, 'height': SIZE},
                         'affineTransform': {'scaleX': RES, 'shearX': 0, 'translateX': x0,
                                             'shearY': 0, 'scaleY': -RES, 'translateY': y0},
                         'crsCode': f'EPSG:{epsg}'}})
            a = np.stack([r['VV'], r['VH']]).astype(np.float32)
            a[~np.isfinite(a)] = np.nan
    except LulcError:
        raise
    except Exception as e:
        raise _translate(e)
    os.makedirs(CACHE, exist_ok=True)
    np.save(p, a)
    return a, n


def fetch_series(lat, lon, start_year, start_month, n_maps=12, log=print):
    lat, lon, start_year, start_month = validate_inputs(lat, lon, start_year, start_month, n_maps)
    out, with_data = [], 0
    for (y, m) in month_list(start_year, start_month, n_maps):
        t = time.time()
        a, n = fetch_month(lat, lon, y, m)
        valid = 100 * np.isfinite(a[0]).mean()
        with_data += int(valid > 0)
        log(f'{y}-{m:02d}: scenes {"cached" if n < 0 else format(n, "6d")} | valid {valid:5.1f}% | VV mean {np.nanmean(a[0]) if valid else float("nan"):6.2f} | {time.time() - t:4.1f} s')
        out.append(a)
    if with_data == 0:
        raise LulcError('No Sentinel-1 scenes (IW, VV+VH) were found for this location in this period. '
                        'Try another start month, or a location with Sentinel-1 coverage (open ocean far from the coast and some polar areas are not imaged in this mode).')
    if with_data < MIN_MONTHS_WITH_DATA:
        log(f'WARNING: only {with_data} of {len(out)} monthly composites have data; maps for the missing months '
            f'will show "no data".')
    return np.stack(out)       # (n_maps + 2, 2, SIZE, SIZE)


if __name__ == '__main__':
    s = fetch_series(1.3521, 103.8198, 2020, 1)
    print('series shape:', s.shape)
