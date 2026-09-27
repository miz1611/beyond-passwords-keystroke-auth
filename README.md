# Beyond Passwords: A Machine Learning Approach to Identity Verification via Keystroke Dynamics

This repository accompanies the paper **"Beyond Passwords: A Machine
Learning Approach to Identity Verification via Keystroke Dynamics"**
(Amruth V, Misbah Anjum G — Dept. of ISE, MIT Mysore).

It implements the paper's proposed pipeline for continuous, behavioural-
biometric authentication based on typing rhythm, using a Random Forest
classifier over Dwell Time, Flight Time, and Distance Enhanced Flight
Time (DEFT) features.

📄 Full paper: [`docs/paper/BEYOND_PASSWORDS.pdf`](docs/paper/BEYOND_PASSWORDS.pdf)
📚 Related work: [`docs/related-works.md`](docs/related-works.md)

## Motivation

Passwords are knowledge-based and only checked once, at login. Keystroke
dynamics instead verify identity continuously, using *how* a person types
rather than *what* they know — a behavioural layer that stays effective
even if the underlying password is compromised, and fits naturally into
Zero Trust authentication models.

## Pipeline

```
Data Acquisition  →  Feature Extraction  →  Classification
 (key press/release)   (dwell, flight,       (Random Forest)
                        DEFT distance)
```

- **Dwell Time** `DT_i = R_i - P_i` — how long a key is held down.
- **Flight Time** `FT_ij = P_j - R_i` — latency between releasing one key
  and pressing the next.
- **DEFT** — flight time combined with the physical (Euclidean) distance
  between the two keys on a QWERTY grid, mapping timing to hand travel
  distance.
- **Random Forest** — chosen for robustness to noisy, non-linear
  behavioural data and its ability to rank feature importance.

## Repository structure

```
beyond-passwords-keystroke-auth/
├── docs/
│   ├── paper/BEYOND_PASSWORDS.pdf   # full paper
│   └── related-works.md             # annotated bibliography
├── src/
│   ├── utils.py                     # QWERTY key-distance lookup
│   ├── feature_extraction.py        # dwell/flight/DEFT feature extraction
│   ├── data_collector.py            # live keystroke capture (pynput)
│   └── train_model.py               # Random Forest training + evaluation
├── data/
│   └── sample_keystroke_data.csv    # synthetic demo dataset
├── results/                         # generated figures (gitignored)
├── requirements.txt
└── LICENSE
```

## Getting started

```bash
git clone https://github.com/<your-username>/beyond-passwords-keystroke-auth.git
cd beyond-passwords-keystroke-auth
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option A — try it instantly on the sample dataset

```bash
cd src
python3 train_model.py --data ../data/sample_keystroke_data.csv
```

This trains the Random Forest classifier and writes three figures to
`results/`: feature importance, the dwell/flight scatter clusters, and
the confusion matrix — reproducing Figures 1–3 from the paper.

### Option B — collect your own keystroke data

```bash
cd src
# capture 10 genuine typing samples for yourself
python3 data_collector.py --user misbah --label 1 --rounds 10

# capture 10 samples from someone else acting as an "imposter"
python3 data_collector.py --user imposter1 --label 0 --rounds 10

# train and evaluate on the data you just collected
python3 train_model.py --data ../data/keystroke_features.csv
```

## Results (paper benchmarks)

| Algorithm       | Accuracy (%) | EER (%) | Best for                     |
|-----------------|-------------|---------|-------------------------------|
| Random Forest   | 98.00       | 4.01    | Fixed-text, edge deployment  |
| XGBoost         | 96.39       | 3.50    | High predictive power        |
| CNN-LSTM        | 98.92       | 2.10    | Continuous free-text         |
| SVM (Linear)    | 97.55       | 2.36    | Robust in high-dim space     |
| LightGBM        | 80.00       | 10.06   | Lightweight, edge devices    |

Random Forest was prioritized in this project for its lower computational
overhead and suitability for local, on-device inference.

## Privacy note

In line with the paper's "Privacy-by-Design" principle, this
implementation runs feature extraction locally — only derived numeric
features (dwell/flight/DEFT statistics), never raw keystroke content, are
ever written to disk or used for classification.

## References & related materials

Full annotated bibliography with notes: [`docs/related-works.md`](docs/related-works.md)

| # | Source | Link |
|---|--------|------|
| 1 | Dillon & Arushi — Agent-based modelling for free-text keyboard dynamics | https://www.researchgate.net/publication/380877463 |
| 2 | Martins et al. — Keystroke dynamics for intelligent biometric authentication with ML (Springer, open access) | https://doi.org/10.1007/s42452-025-07449-5 |
| 3 | Dhakal et al. — Observations on Typing from 136 Million Keystrokes (ACM) | https://doi.org/10.1145/3173574.3174220 |
| 4 | The Guardian — Bank Sepah "Codebreakers" breach | https://www.theguardian.com/world/2025/mar/bank-sepah-cyber-attack/ |
| 5 | TechCrunch — PowerSchool data breach | https://techcrunch.com/2025/01/powerschool-breach-60-million/ |
| 6 | IBM — Cost of a Data Breach Report 2025 | https://www.ibm.com/reports/data-breach-cost |
| 7 | Identity Theft Resource Center — 2025 Trends report | https://www.idtheftcenter.org/publications/2025-identity-theft-report/ |
| 8 | BioCatch — TrickBot detection case study | https://www.biocatch.com/resources/case-studies/a-top-5-u.s.-bank-detects-trickbot-malware-attacks-with-biocatchs-behavioral-biometrics-solution |
| 9 | LexisNexis / BehavioSec — continuous authentication for Global 2000 | https://risk.lexisnexis.com/global/en/products/behaviosec |
| 10 | BioCatch — Top LATAM bank, 66% false-positive reduction | https://www.biocatch.com/hubfs/White%20Papers/New%20Brand/Biocatch_CS_Top_LATAM_DetecttoPrevent.pdf |

## Author

**Misbah Anjum G** — B.E. Information Science & Engineering, Maharaja
Institute of Technology, Mysore (VTU). [LinkedIn](https://linkedin.com/in/misbahanjum)

## License

MIT — see [`LICENSE`](LICENSE).
