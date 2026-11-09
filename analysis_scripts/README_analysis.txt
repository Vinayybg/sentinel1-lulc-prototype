Analysis scripts used for the report. They are NOT needed to run the prototype.

Run them from the top folder (the one with app.py), with the top folder on the Python path, e.g. on Windows:
  set PYTHONPATH=.
  python analysis_scripts\\case_study_singapore.py
macOS / Linux:
  PYTHONPATH=. python analysis_scripts/case_study_singapore.py

Extra packages: pip install -r analysis_scripts/requirements-casestudy.txt
The Singapore and CORINE case studies need reference data (URA Master Plan, OpenStreetMap, CORINE) in a folder
'reference/'; build_reference.py and get_osm_singapore.py download and prepare it. Results from the report runs are in the
report appendix and the figures of the report. Requires the weights and an Earth Engine login as described in the main README.
