import numpy as np, geopandas as gpd, pyproj
from rasterio import features
from rasterio.transform import Affine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from fetch_s1 import grid_for, SIZE, RES

LAT, LON = 1.3521, 103.8198
epsg, x0, y0 = grid_for(LAT, LON)
tf = Affine(RES, 0, x0, 0, -RES, y0)
tr = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
lo0, la0 = tr.transform(x0 - 500, y0 - SIZE * RES - 500)
lo1, la1 = tr.transform(x0 + SIZE * RES + 500, y0 + 500)
bbox = (lo0, la0, lo1, la1)

WATER, TREES, LOW, BUILT, BARE, GREEN, IGN = 0, 1, 2, 3, 4, 6, 255
BUILT_LU = ['RESIDENTIAL', 'ROAD', 'COMMERCIAL', 'BUSINESS 2', 'RESIDENTIAL WITH COMMERCIAL AT 1ST STOREY',
            'BUSINESS 1', 'UTILITY', 'RESIDENTIAL / INSTITUTION', 'COMMERCIAL & RESIDENTIAL', 'PLACE OF WORSHIP',
            'CIVIC & COMMUNITY INSTITUTION', 'EDUCATIONAL INSTITUTION', 'COMMERCIAL / INSTITUTION',
            'TRANSPORT FACILITIES', 'HOTEL', 'BUSINESS PARK', 'HEALTH & MEDICAL CARE', 'WHITE', 'SPECIAL USE',
            'PORT / AIRPORT', 'MASS RAPID TRANSIT', 'BUSINESS 1 - WHITE', 'LIGHT RAPID TRANSIT',
            'BUSINESS 2 - WHITE', 'BUSINESS PARK - WHITE']
GREEN_LU = ['PARK', 'OPEN SPACE', 'RESERVE SITE', 'CEMETERY', 'SPORTS & RECREATION']


def lu_code(d):
    if d in BUILT_LU: return BUILT
    if d in GREEN_LU: return GREEN
    if d == 'AGRICULTURE': return LOW
    if d == 'WATERBODY': return WATER
    if d == 'BEACH AREA': return BARE
    return IGN


def shapes_of(gdf, code):
    gdf = gdf.to_crs(epsg)
    return [(g, code) for g in gdf.geometry.make_valid() if g is not None and not g.is_empty]


ref = np.full((SIZE, SIZE), IGN, np.uint8)

ura = gpd.read_file('reference/ura/MasterPlan2019LandUselayer.geojson', bbox=bbox)
print('URA parcels in the area:', len(ura))
ura = ura.to_crs(epsg)
sh = [(g, lu_code(d)) for g, d in zip(ura.geometry.make_valid(), ura['LU_DESC']) if g is not None and not g.is_empty]
features.rasterize(sh, out=ref, transform=tf)

gp = 'reference/osm/singapore_osm.gpkg'
for layer, code in [('trees', TREES), ('grass', LOW), ('crops', LOW), ('water', WATER), ('buildings', BUILT), ('scrub_bare', IGN)]:
    g = gpd.read_file(gp, layer=layer, bbox=bbox)
    print('OSM', layer, ':', len(g), 'polygons')
    if len(g):
        features.rasterize(shapes_of(g, code), out=ref, transform=tf)

np.save('cache/ref_singapore_2020.npy', ref)
names = {0: 'water', 1: 'trees', 2: 'low_veg', 3: 'built', 4: 'bare', 6: 'green (unspecified)', 255: 'not scored'}
print()
for c, n in names.items():
    print(f'{n:22s} {100 * (ref == c).mean():6.2f} %')

COL = {0: '#2b6cb0', 1: '#1f6b2e', 2: '#8bd35f', 3: '#e53e3e', 4: '#d2b48c', 5: '#f7fafc', 6: '#b7e4a8', 255: '#2d2d2d'}
lut = np.zeros((256, 3), np.uint8)
for c, h in COL.items():
    lut[c] = [int(h[i:i + 2], 16) for i in (1, 3, 5)]
m9 = np.load('cache/maps_singapore_2020.npy')[0]
to6 = np.array([0, 1, 2, 2, 2, 2, 3, 4, 5] + [255] * 247, np.uint8)
fig, ax = plt.subplots(1, 2, figsize=(14, 7.4))
ax[0].imshow(lut[ref][::2, ::2]); ax[0].set_title('Reference: URA Master Plan 2019 + OpenStreetMap')
ax[1].imshow(lut[to6[m9]][::2, ::2]); ax[1].set_title('Model, 2020-01 (merged to 6 classes)')
for a in ax: a.axis('off')
hs = [Patch(color=COL[c], label=n) for c, n in names.items()]
fig.legend(handles=hs, loc='lower center', ncol=7, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig('outputs/singapore_reference_check.png', dpi=100)
print('saved outputs/singapore_reference_check.png')
