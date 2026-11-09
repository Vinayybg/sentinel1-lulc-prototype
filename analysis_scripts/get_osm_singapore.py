"""Download OpenStreetMap land-cover related polygons for the Singapore case-study tile.
Output: reference/osm/singapore_osm.gpkg (one layer per class group).
Data (c) OpenStreetMap contributors, ODbL. https://www.openstreetmap.org/copyright
"""
import math, os
import osmnx as ox
import geopandas as gpd
import pandas as pd

LAT, LON, HALF_M = 1.3521, 103.8198, 5000          # 10 km x 10 km tile centred on this point
dlat = HALF_M / 111320.0
dlon = HALF_M / (111320.0 * math.cos(math.radians(LAT)))
BBOX = (LON - dlon, LAT - dlat, LON + dlon, LAT + dlat)   # (left, bottom, right, top)
print("bbox (left, bottom, right, top):", [round(v, 5) for v in BBOX])

OUT_DIR = os.path.join("reference", "osm")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "singapore_osm.gpkg")
if os.path.exists(OUT):
    os.remove(OUT)

ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.timeout = 300

LAYERS = {
    "buildings": {"building": True},
    "water":     {"natural": ["water", "wetland"], "waterway": "riverbank", "landuse": ["reservoir", "basin"]},
    "trees":     {"natural": "wood", "landuse": "forest"},
    "grass":     {"landuse": ["grass", "meadow", "recreation_ground", "village_green", "cemetery"],
                  "leisure": ["park", "garden", "pitch", "golf_course", "playground"]},
    "crops":     {"landuse": ["farmland", "farmyard", "orchard", "vineyard", "greenhouse_horticulture", "allotments"]},
    "scrub_bare": {"natural": ["scrub", "heath", "bare_rock", "sand", "beach"]},
}

for name, tags in LAYERS.items():
    print(f"downloading {name} ...", flush=True)
    try:
        gdf = ox.features_from_bbox(BBOX, tags)
    except Exception as e:                     # empty result raises an error in some osmnx versions
        print(f"  {name}: nothing returned ({type(e).__name__}: {e})")
        continue
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf[["geometry"]].reset_index(drop=True)
    gdf.to_file(OUT, layer=name, driver="GPKG")
    print(f"  {name}: {len(gdf)} polygons")

print("done ->", OUT, "| size MB:", round(os.path.getsize(OUT) / 1e6, 1))
