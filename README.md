# Fitness Tracker Machine Learning Project

This project is a machine learning pipeline for analyzing and classifying fitness activity data collected from wearable sensors. The workflow includes data preprocessing, feature engineering, model training, and evaluation, with the goal of recognizing and analyzing different exercise types and repetitions.

## Project Structure

```
├── data/
│   ├── external/           # External data sources
│   ├── interim/            # Intermediate data (pickles, processed steps)
│   ├── processed/          # Final processed datasets
│   └── raw/                # Raw sensor data (CSV files)
├── docs/                   # Documentation
├── models/                 # Saved models
├── notebooks/              # Jupyter notebooks for exploration
├── references/             # Reference materials
├── reports/
│   └── figures/            # Generated figures and plots
├── src/
│   ├── data/               # Data loading and preprocessing scripts
│   ├── features/           # Feature engineering scripts
│   ├── models/             # Model training and evaluation scripts
│   └── visualization/      # Visualization scripts
├── environment.yml         # Conda environment definition
├── requirements.txt        # Python dependencies
└── README.md               # Project overview (this file)
```

## Main Features
- **Data Preprocessing:** Scripts to clean, transform, and prepare sensor data.
- **Feature Engineering:** Extraction of time, frequency, and statistical features from raw signals.
- **Model Training:** Multiple machine learning models (Random Forest, Neural Network, KNN, etc.) for activity classification.
- **Evaluation:** Visualization and comparison of model performance.
- **Repetition Counting:** (Optional) Scripts to count exercise repetitions from sensor data.

## Getting Started

### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/fitness-tracker-ml.git
cd fitness-tracker-ml
```

### 2. Set Up the Environment
Using Conda:
```sh
conda env create -f environment.yml
conda activate fitness-tracker
```
Or using pip:
```sh
pip install -r requirements.txt
```

### 3. Data
- Raw sensor data is already present in the `data/raw/MetaMotion/` directory.
- Intermediate and processed data will be saved in `data/interim/` and `data/processed/`.

### 4. Running the Pipeline
- Data processing: `src/data/make_dataset.py`
- Feature engineering: `src/features/build_features.py`
- Model training: `src/models/train_model.py`
- Visualization: `src/visualization/visualize.py`

You can run these scripts directly or use the provided Jupyter notebooks for step-by-step exploration.

## Results
- Model performance metrics and confusion matrices are saved in `reports/figures/`.
- Trained models are saved in the `models/` directory.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.