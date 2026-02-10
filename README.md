# Turbulence AE Index Forecasting

This repository contains scripts to construct and train multiple XGBoost models to forecast AE index at Earth 🌍 given OMNI data measured at L1.

## 🔧 Installation
Ensure Python is already installed on the system. Python 3.12.12 was used for this project, but lower (or higher) versions may work.

With HTTPS:
```bash
git clone https://github.com/carawaters/TurbulenceSpaceWeatherForecast.git
cd TurbulenceSpaceWeatherForecast
pip install -r requirements.txt
```

With SSH:
```bash
git clone git@github.com:carawaters/TurbulenceSpaceWeatherForecast.git
cd TurbulenceSpaceWeatherForecast
pip install -r requirements.txt
```

## ▶️ Usage
```bash
python {base/turbulence/noise}_model.py
```

These scripts train the respective models at different forecast horizons using OMNI data.

## 📁 Project Structure
```
├── model_ae_xgb_base_new/            # Base forecast models and metadata
├── models_ae_xgb_noise_new/          # Noise forecast models and metadata
├── models_ae_xgb_turb_new/           # Turbulence forecast models and metadata
├── base_model.py                     # Script to train base model
├── noise_model.py                    # Script to train noise model
├── turbulence_model.py               # Script to train turbulence model
├── xgb_optuna_results_base_new.csv   # Summary of different horizon training outcomes for base model
├── xgb_optuna_results_noise_new.csv  # Summary of different horizon training outcomes for noise model
└── xgb_optuna_results_turb_new.csv   # Summary of different horizon training outcomes for training model
```

## 🐍 Requirements
- Python 3.12.12
- PySPEDAS
- XGBoost
- Optuna
- Scikit-learn
- See `requirements.txt` for full dependencies and package versions

## 👩‍💻 Author
Cara Waters, 2026
Queen Mary University of London
For questions, contact: c.waters@qmul.ac.uk
