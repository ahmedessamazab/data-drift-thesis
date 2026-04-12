# PNN Experiment: Concept Drift Detection using Incremental Probabilistic Neural Networks

## Overview

This project implements and evaluates an Incremental Probabilistic Neural Network (IPNN) for concept drift detection in data streams. The IPNN uses orthonormal series (Hermite polynomials) for density estimation in non-stationary environments.

The system consists of:
- **Synthetic Data Generator**: Creates data streams with multiple types of concept drifts (mean, variance, gradual, cyclic, distribution changes)
- **IPNN Model**: Incremental density estimation using orthogonal series expansion
- **Drift Detector**: Monitors Integrated Squared Error (ISE) between reference and current probability density functions
- **Experiment Framework**: Comprehensive evaluation with visualization and CSV exports


## Features

### Drift Types Supported
- **Mean Drift**: Sudden shift in distribution mean
- **Variance Drift**: Change in standard deviation
- **Gradual Drift**: Smooth transition over a window
- **Cyclic Drift**: Periodic sinusoidal changes
- **Distribution Drift**: Structural changes (e.g., Gaussian → Uniform)

### Key Components
- `ipnn.py`: Incremental Probabilistic Neural Network implementation
- `drift_detector.py`: ISE-based drift detection algorithm
- `synthetic_stream.py`: Multi-drift synthetic data generator
- `run_full_experiment.py`: Complete experiment runner with visualization
- `series.py`: Orthogonal series implementations (Hermite, Legendre, etc.)

## Installation

1. Clone or download this repository
2. Create a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Requirements
- Python 3.7+
- numpy
- matplotlib
- pandas

## Usage

### Running a Full Experiment

Execute the main experiment script:

```bash
python run_full_experiment.py
```

This will:
- Generate a synthetic stream with 8 different drift types
- Run IPNN-based drift detection
- Produce plots and CSV files in a timestamped experiment folder

### Generated Outputs

**Plots:**
- `01_stream_overview.png`: Raw data stream with drift annotations
- `02_detection_report.png`: ISE scores and detection results
- `03_pdf_evolution.png`: PDF evolution over time

**CSV Files:**
- `csv_01_stream_raw.csv`: Raw stream data with segment info
- `csv_02_drift_ground_truth.csv`: Injected drift events
- `csv_03_ise_score.csv`: ISE scores over time
- `csv_04_coefficients.csv`: Hermite coefficients evolution
- `csv_05_detection_results.csv`: Detection accuracy report
- `csv_06_pdf_snapshots.csv`: PDF curves at key moments

### Configuration

Experiments are configured via parameters in `run_full_experiment.py`:
- `N`: Stream length (default: 10,000)
- `Q_PARAM`, `K_PARAM`, `GAMMA`: IPNN hyperparameters
- `WARMUP`: Initial samples for reference PDF
- `ISE_THRESHOLD`: Drift detection threshold
- Drift specifications in the `drifts` list

## Project Structure

```
PNN_experiment/
├── ipnn.py                    # IPNN implementation
├── drift_detector.py          # Drift detection logic
├── synthetic_stream.py        # Synthetic data generator
├── run_full_experiment.py     # Main experiment script
├── series.py                  # Orthogonal series functions
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── experiments/               # Experiment outputs
│   └── [timestamp]_.../       # Individual experiment folders
└── myenv/                     # Virtual environment (created)
```

## Methodology

1. **Density Estimation**: IPNN uses Hermite polynomial expansion for PDF estimation
2. **Incremental Learning**: Model adapts to new data with decaying learning rate
3. **Drift Detection**: Compares current PDF against reference PDF using ISE
4. **Evaluation**: Ground truth vs. detected drifts with precision/recall metrics

## Results

The experiments demonstrate the IPNN's ability to detect various concept drift types in streaming data. Performance metrics include:
- True Positive Rate (TPR)
- False Positive Rate (FPR)
- Detection delay
- ISE threshold sensitivity

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Ahmed Essam Azab

## Academic Context

This implementation is part of a Master's thesis research on machine learning for non-stationary data streams, focusing on adaptive density estimation and concept drift detection algorithms.