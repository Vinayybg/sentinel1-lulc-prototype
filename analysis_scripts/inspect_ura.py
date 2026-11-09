import re
import geopandas as gpd

g = gpd.read_file('reference/ura/MasterPlan2019LandUselayer.geojson')
print('rows:', len(g), '| crs:', g.crs)
print('columns:', list(g.columns))
if 'LU_DESC' in g.columns:
    col = g['LU_DESC']
else:
    dcol = 'Description' if 'Description' in g.columns else g.columns[0]
    print('sample:', str(g[dcol].iloc[0])[:400])
    col = g[dcol].astype(str).str.extract(r'LU_DESC</th>\s*<td>(.*?)</td>', flags=re.S)[0]
print()
print(col.value_counts().to_string())
