import math, time, osmnx as ox
LAT, LON, HALF_M = 1.3521, 103.8198, 5000
dlat = HALF_M / 111320.0
dlon = HALF_M / (111320.0 * math.cos(math.radians(LAT)))
BBOX = (LON - dlon, LAT - dlat, LON + dlon, LAT + dlat)
OUT = "reference/osm/singapore_osm.gpkg"
tags = {"natural": ["scrub", "heath", "bare_rock", "sand", "beach"]}
gdf = None
for attempt in range(1, 6):
    try:
        gdf = ox.features_from_bbox(BBOX, tags)
        break
    except Exception as e:
        print("attempt", attempt, "failed:", str(e)[:120].replace("\n", " "))
        if "nothing" in str(e).lower() or "insufficient" in type(e).__name__.lower():
            break
        time.sleep(30 * attempt)
if gdf is None:
    print("no scrub/bare polygons saved")
else:
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])][["geometry"]].reset_index(drop=True)
    gdf.to_file(OUT, layer="scrub_bare", driver="GPKG")
    print("scrub_bare:", len(gdf), "polygons saved")
import pyogrio
print(pyogrio.list_layers(OUT))
