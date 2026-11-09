#!/usr/bin/env bash
# One-time setup (macOS / Linux). Needs Python 3.11 (python3 --version).
cd "$(dirname "$0")"
python3 --version || { echo "Python 3 not found. Install Python 3.11."; exit 1; }
if ! python3 -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)"; then
  echo "WARNING: this project was tested with Python 3.11 only. Other versions may fail to install the pinned packages."
  read -p "Continue anyway? Type y to continue: " CONT
  [ "$CONT" = "y" ] || exit 1
fi
echo "[1/3] Creating the environment and installing packages (several minutes)..."
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt || { echo "Installation failed. See README section 3."; exit 1; }
echo
echo "[2/3] Google Earth Engine project"
echo "Enter the ID of your Google Cloud project that has the Earth Engine API enabled (README section 2)."
read -p "Project ID: " EEP
echo "$EEP" > ee_project.txt
echo
echo "[3/3] Earth Engine login: a browser window opens. Sign in and allow access."
.venv/bin/earthengine authenticate
echo
echo "Setup finished. Start the app with ./run_app.sh"
