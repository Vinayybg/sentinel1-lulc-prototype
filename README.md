# Sentinel-1 land use / land cover (LULC) prototype

Enter a latitude, longitude and a start month. The prototype downloads Sentinel-1 radar data for that place
from Google Earth Engine, predicts a land-cover map for each of 12 consecutive months, and reports
per-month class percentages, month-to-month change and first-to-last change in percentage points,
plus a 12-month visualization. The model uses **Sentinel-1 radar only** (no optical data, no labels at run time).

## Quick start (3 steps)

1. Have ready: Python 3.11, a Google account registered for Google Earth Engine, and a Google Cloud project with the Earth Engine API enabled (section 2 explains how, about 5 minutes).
2. Run **setup.bat** (Windows) or **./setup.sh** (macOS/Linux) once. It installs everything, asks for your Cloud project ID, and opens the Earth Engine login in your browser.
3. Run **run_app.bat** (or **./run_app.sh**). The app opens in your browser: enter latitude, longitude and a start month, press *Run analysis*.

Manual setup (conda, command line, other options) is described in sections 3 to 6.

## 1. What is in this folder

| File / folder | Purpose |
|---|---|
| `app.py` | Streamlit web app (main prototype) |
| `run_lulc.py` | Same pipeline from the command line |
| `fetch_s1.py` | Earth Engine download of monthly Sentinel-1 composites, input checks, cache |
| `lulc_model.py` | Model registry, loading, whole-area prediction, ensemble, class reliability |
| `lulc_report.py`, `lulc_smooth.py`, `lulc_stats.py` | Percentages, change tables, 3-month smoothing, persistent change, trend test |
| `weights/` | Trained model weights (`*.pt`) and `s1_norm.json` (normalisation constants) |
| `cache/` | Downloaded composites (created automatically; safe to delete) |
| `outputs/` | Results of each run (created automatically) |
| `setup.bat`, `setup.sh`, `run_app.bat`, `run_app.sh` | one-time setup and app start |
| `analysis_scripts/` | Scripts used for the report (case studies Singapore and CORINE, ensemble/confusion evaluation, noise floor, SAR signatures). Not needed to run the prototype; see `analysis_scripts/README_analysis.txt` |
| `test_models.py` | Quick self-test of all models |

## 2. Accounts you need (one-time, about 5-10 minutes)

The prototype downloads Sentinel-1 data from Google Earth Engine (free for non-commercial and academic use). No key file is stored in this folder; you use your own Google account.

1. **Google account** (any personal or university account).
2. **Register for non-commercial Earth Engine use and create a Cloud project:**
   1. Open https://code.earthengine.google.com/register and sign in with the Google account.
   2. Choose **Unpaid usage** (non-commercial), then the purpose that fits, for example **Academia / education** or a **college or university project**.
   3. **Create a new Cloud project** when asked (any name, for example `lulc-test`) and note the **project ID** shown, for example `lulc-test-123456`. You need this ID for setup.
   4. Answer the short eligibility questions (organisation or university name, a one-sentence description of the use, for example "land-cover classification from Sentinel-1 radar for a university course project"). Eligible academic projects are confirmed immediately.
3. **Check that the Earth Engine API is enabled** for that project: open https://console.cloud.google.com/apis/library, select your project at the top, search for **Google Earth Engine API** and press **Enable** if it is not enabled yet. (Registration normally does this for you.)
4. **Log in once** from the computer that runs the prototype: `earthengine authenticate` (the setup script does this for you).

Check your usage at any time at https://console.cloud.google.com/earth-engine (project selected at the top, "Monthly EECU-second usage").

## 3. Install (Python 3.11, CPU only, no GPU needed)

```
conda create -n lulc python=3.11 -y
conda activate lulc
pip install -r requirements.txt
```

## 4. Earth Engine setup (one-time login)

```
earthengine authenticate
```
A browser window opens; sign in and allow access. Then tell the program your project ID.

Windows PowerShell:
```
$env:EE_PROJECT = "your-cloud-project-id"
```
macOS / Linux:
```
export EE_PROJECT="your-cloud-project-id"
```
(`GOOGLE_CLOUD_PROJECT` is also accepted. If you ran `setup.bat` / `setup.sh`, the project ID is stored in `ee_project.txt` and no variable is needed.) Set this in the same terminal before starting the app.

## 5. Weights

The trained weights are published as a release asset: **https://github.com/Vinayybg/sentinel1-lulc-prototype/releases/tag/v1.0**
Download **weights_all_models.zip** (about 300 MB), unzip it, and copy all files into the `weights/` folder of this project.
`E4_consensus_unet_pre_best.pt` is the final model; the other four are alternatives for the *Model* dropdown and the ensemble; `s1_norm.json` holds the normalisation constants.
The app runs with the final model plus `s1_norm.json` alone.
Direct link: https://github.com/Vinayybg/sentinel1-lulc-prototype/releases/download/v1.0/weights_all_models.zip

## 6. Run

Web app:
```
streamlit run app.py
```
Command line:
```
python run_lulc.py --lat 1.3521 --lon 103.8198 --year 2020 --month 1
python run_lulc.py --lat 49.4875 --lon 8.466 --year 2020 --month 1 --model ensemble --smooth
```
`--model` accepts `unet_pre` (default, final model), `unet_scratch`, `deeplab_pre`, `deeplab_scratch`, `fpn_r50_pre`, `ensemble`.
`--smooth` applies a 3-month majority vote to the maps.

Inputs: latitude -84..84, longitude -180..180, start month no earlier than 2014-11. The last of the 12 monthly
maps must be a completed past month. The mapped area is about 10 km x 10 km around the point.

## 7. Outputs (folder `outputs/<lat>_<lon>_<start>_<model>/`)

| File | Content |
|---|---|
| `maps_12_months.png` | the 12 monthly land-cover maps with legend |
| `class_percentages.csv` | share of each class (%) per month |
| `change_month_to_month_pp.csv` | change in percentage points between consecutive months |
| `change_first_to_last_pp.csv` | change in percentage points from the first to the last month |
| `change_significance.csv` | 12-month trend test per class (Theil-Sen slope, Mann-Kendall p; "trend" only if p<0.05 and the change exceeds twice the monthly noise and 0.5 pp) |
| `persistent_change_summary.csv` | share of pixels that changed class between the first 3 and last 3 months (majority vote), compared with the raw first-vs-last change |
| `persistent_change_per_class.csv` | per class: start and end share, net change, pixels that left / entered |
| `persistent_change_transitions.csv` | the largest class-to-class transitions (% of the area) |

The web app shows the same tables and figures and offers them as downloads.

## 8. Run time and Earth Engine cost

- First run for a place: about 1-3 minutes (12 + 2 monthly downloads); repeated runs use `cache/` and take seconds.
- Prediction on CPU: about 2 s per map (ensemble about 13 s per map).
- A run uses roughly 0.03 % of the free monthly Earth Engine quota (Community tier: 150 EECU-hours per month, resets on the 1st of each month). If a project has used its quota, Earth Engine switches to "restricted mode": runs still work, only slower. Months already downloaded are kept in `cache/` and never downloaded again.

## 9. Model, accuracy and known limits

Final model: U-Net with ImageNet-pretrained ResNet-18, 9 classes, input = 3 consecutive months x VV/VH.
Test set (147 places, 4,402 patches): OA 0.799, mIoU 0.623. Ensemble of 5 models: OA 0.821, mIoU 0.653.
Labels are a consensus of Dynamic World, ESA WorldCover and ESRI Land Cover, so the model reproduces
that consensus, not ground truth.

Known limits:
- Radar cannot separate some classes well: shrub vs trees vs grass (IoU 0.35 / 0.80 / 0.47), snow (0.40), flooded vegetation (0.52).
- Dense city areas may be over-estimated as built-up, and low vegetation inside cities is under-detected.
- Month-to-month "change" includes noise (seasonality, soil moisture, wet snow, speckle); small percentage-point changes are often not real. Use the trend test and persistent-change tables.
- Accuracy varies by region; the test set has few places per continent, so continent scores are indicative only.
- Single models are over-confident; the ensemble is well calibrated.

## 10. Troubleshooting

| Message | Fix |
|---|---|
| "EE_PROJECT not set" | Set the environment variable in step 4 in the same terminal |
| "Please authenticate" | Run `earthengine authenticate` |
| "Permission denied / project not registered" | Enable Earth Engine API on the Cloud project and register it for Earth Engine |
| "Weights not found" | Put the `.pt` files in `weights/` (section 5) |
| "No Sentinel-1 scenes" | Place or period has no data (e.g. before late 2014, or at extreme latitudes); try another month |
| Quota / network errors | Wait and retry; cached months are not downloaded again |

## 11. Data and licences

Sentinel-1 (Copernicus, via Earth Engine), Dynamic World (CC BY 4.0), ESA WorldCover (CC BY 4.0), ESRI Land Cover (CC BY 4.0),
URA Master Plan 2019 and OpenStreetMap (ODbL) for the Singapore case study.

## 12. Training data and notebooks (Kaggle)

The training, validation and test arrays are too large to ship (several GB). They are the output files of these public Kaggle notebooks
(open the notebook, tab **Output**, download; or `kaggle kernels output vinayybg/<name> -p <folder>`):

| What | Notebook (https://www.kaggle.com/code/vinayybg/NAME) |
|---|---|
| Training part 1 + validation arrays | `v2-02-subsets` |
| Training part 2 arrays | `v2-02b-subsets` |
| Training part 3 arrays | `v2-02c-subsets` |
| Test arrays | `v2-03c-test-subset` |
| Place list | dataset https://www.kaggle.com/datasets/vinayybg/places-v2 |

Executed notebooks (with training curves and results) are also in the `notebooks/` folder of the submission:
radar download `v2-run01`-`v2-run05`, label audit `v2-00-audit`, consensus `v2-01-consensus`, data subsets `v2-02*`, `v2-03c-test-subset`,
training `v2-03-train-*` (nine runs), evaluation `v2-04-test-eval`, `v2-04b-test-eval-all-models`, `v2-05-ensemble-uncertainty`.
The full list with descriptions is in the appendix of the report.
