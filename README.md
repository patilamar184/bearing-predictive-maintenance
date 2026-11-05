# Bearing Predictive Maintenance — Anomaly Detection

A Python-based machine-learning prototype that analyses vibration measurements from the NASA IMS bearing dataset and identifies recordings with unusual vibration patterns.

> **Project status:** Initial anomaly-detection prototype. This project currently detects unusual vibration patterns; it does **not** establish that a bearing will fail, predict an exact failure date, or estimate remaining useful life (RUL).

## Problem statement

Bearings are critical components in rotating machinery. Changes in vibration behaviour can indicate changes in machine condition. This project explores how vibration measurements can be transformed into statistical features and analysed with an unsupervised machine-learning model.

## Current capabilities

- Loads individual vibration recordings from the NASA IMS dataset.
- Extracts statistical features from each vibration channel.
- Uses a scikit-learn `Pipeline` containing `StandardScaler` and `IsolationForest`.
- Scores a recording as `NORMAL` or `ANOMALY`.
- Saves the trained model with Joblib for reuse.

## Technology stack

- Python
- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib and Seaborn (notebook exploration)
- Joblib
- Google Colab

## Repository structure

```text
bearing-predictive-maintenance/
├── models/
│   ├── bearing_anomaly_model.joblib
│   └── feature_columns.json
├── notebooks/
│   └── Bearing_Predictive_Maintenance_ML.ipynb
├── src/
│   ├── feature_extraction.py
│   └── predict.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

This project uses the **NASA IMS Bearing Dataset**, which contains vibration measurements collected from bearing test rigs during run-to-failure experiments.

Dataset source:
- [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

The initial experiment used in the notebook contains 2,156 timestamped recordings. The raw dataset is not included in this repository. Download it from the source above and follow the notebook's data-loading instructions.

The dataset contains multiple vibration channels. The notebook extracts eight statistical features per channel:

- Mean
- Standard deviation
- Variance
- Root mean square (RMS)
- Skewness
- Kurtosis
- Peak-to-peak amplitude
- Crest factor

For the first experiment, eight channels produce 64 features per recording.

## Model approach

The current prototype uses **Isolation Forest**, an unsupervised anomaly-detection algorithm.

The notebook uses the earliest 20% of the ordered recordings as an assumed baseline for model fitting. This is an experimental assumption, not a verified healthy-state label. The model then scores recordings according to how unusual their feature patterns are relative to the fitted baseline.

Interpretation:
- `1` — NORMAL (not identified as an anomaly by the model)
- `-1` — ANOMALY (identified as an unusual pattern)

The `decision_function` score is also returned. A lower score indicates a more anomalous observation relative to this fitted model.

**Important limitations**
- An anomaly is not automatically a confirmed bearing fault.
- The initial baseline assumption has not been independently verified.
- This prototype has not yet been validated on a separate bearing experiment.
- The current dataset does not provide ready-made labels for every recording.
- No remaining-useful-life estimate is produced.
- Temperature and RPM are not inputs to this model; the current model uses vibration-derived features only.

Do not use this prototype to make safety-critical or production maintenance decisions.

## Setup

Use Python 3.10+ (or a compatible environment).

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/bearing-predictive-maintenance.git
   cd bearing-predictive-maintenance
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the NASA IMS dataset and extract a recording file locally. The raw dataset is intentionally excluded from Git.

## Run a prediction

The prediction script accepts the path to one raw, whitespace-separated vibration recording:

```bash
python src/predict.py path/to/recording_file
```

The script expects these model artifacts to exist:

```text
models/bearing_anomaly_model.joblib
models/feature_columns.json
```

Example output format:

```text
Recording:     <recording_filename>
Status:        NORMAL
Prediction:    1  (1=normal, -1=anomaly)
Anomaly score: <calculated_score>
```

The output above illustrates the format only; it is not a measured prediction.

## Reproducing the notebook

Open `notebooks/Bearing_Predictive_Maintenance_ML.ipynb` in Google Colab or Jupyter.

The notebook covers:
1. Environment setup
2. Dataset loading and inspection
3. Exploratory data analysis
4. Vibration feature extraction
5. Isolation Forest training
6. Anomaly scoring and visualisation
7. Model export

After training, export both the model and its feature-column list into the `models/` directory. Keep the feature extraction logic consistent between training and inference.

## Roadmap

- Validate the baseline assumption using experiment documentation and domain knowledge.
- Evaluate on a separate bearing experiment to reduce data leakage.
- Compare anomaly-detection approaches and tune thresholds using a documented validation strategy.
- Add automated tests for feature extraction and inference.
- Build a FastAPI service around the prediction module.
- Add persistent machine and sensor-reading storage with PostgreSQL.
- Explore labelled failure prediction and remaining-useful-life estimation only when suitable labelled data and validation are available.
- Add GenAI-generated maintenance explanations as a separate application layer, clearly distinguishing generated recommendations from measured model outputs.

## Author

**<Your Name>**

Add your GitHub profile and project links here.

## Disclaimer

This repository is an educational and portfolio project. It is not an official Schaeffler product and is not affiliated with or endorsed by Schaeffler or NASA.
