# ml4hlt

Bayesian ML analysis of AMBER DAQ feature data for high-level trigger filtering.

## Setup

```bash
git clone
cd ml4hlt

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project structure

```
ml4hlt/
├── data/               # ROOT files from FeatureExtractor
├── plots/              # Output plots
├── scripts/
│   └── explore.py      # Exploratory analysis
│   └── train.py
│   └── stitch.py
└── src/
    ├── config.py       # Paths and physical constants
    ├── features/
    │   └── loader.py   # ROOT → pandas DataFrames
    │   └── builder.py
    ├── models/         # Bayesian models
    │   └── gaussian_classifier.py
    └── utils/          # Shared helpers
        ├── preprocessing.py
        └── evaluation.py
```

## Usage

```bash
source venv/bin/activate

python scripts/explore.py data/amber-merged-002406-00010-00.features.root
```

Produces four PDFs in `plots/`:
- `trigger_offset.pdf`         — TPC trigger arrival distribution within timeslice
- `per_source_profiles.pdf`    — mean hit count vs image index per source, triggered vs not
- `trigger_aligned_profiles.pdf` — same but time axis aligned to trigger position
- `feature_distributions.pdf`  — per-feature density: triggered vs non-triggered