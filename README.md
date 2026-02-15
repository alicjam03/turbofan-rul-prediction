# Turbofan Engine Remaining Useful Life (RUL) Prediction

## Overview

This project investigates **Remaining Useful Life (RUL)** prediction for turbofan engines using the NASA CMAPSS FD001 dataset.

The objective is to compare classical machine learning and deep learning approaches for predictive maintenance, with a particular focus on safety-critical error behaviour (minimising dangerous overprediction near engine failure).

---

## Dataset: CMAPSS FD001

- Train trajectories: 100  
- Test trajectories: 100  
- Operating conditions: 1 (Sea Level)  
- Fault modes: 1 (High Pressure Compressor degradation)

FD001 represents a controlled degradation scenario (single condition, single fault mode), enabling structured model comparison.

---

## Problem Framing

Remaining Useful Life (RUL) was computed for the training set as:

RUL = (final cycle of engine) − (current cycle)

The official NASA train/test split was respected throughout to prevent data leakage.

---

## Exploratory Data Analysis

Key preprocessing and analysis steps:

- Removed columns containing no useful information.
- Identified and dropped sensors with near-zero variance.
- Analysed variance and correlation between sensors and RUL.
- Visualised degradation trajectories across multiple engines.
- Confirmed consistent degradation behaviour across units.
- Observed acceleration in degradation near failure.

Key findings:

- Several sensors increase as RUL decreases.
- Some sensors decrease as RUL decreases.
- Degradation patterns were consistent across engines.

These findings informed feature engineering choices.

---

## Feature Engineering

A sliding window approach was applied using a window size of 30 cycles.

For each window, the following features were computed:

- Mean  
- Standard deviation  
- Minimum  
- Maximum  

The test dataset was processed identically to ensure consistency.

This transformed the time-series problem into a structured regression task suitable for tree-based models.

---

## Classical Machine Learning Models

### Random Forest (Baseline – Uncapped RUL)

- MAE: 15.09  
- RMSE: 21.51  

Observations:

- Strong performance near failure.
- Larger overprediction at higher RUL values.

Since overprediction is more dangerous in safety-critical systems, error behaviour was analysed beyond global metrics.

---

### XGBoost

Multiple learning rates were tested:

| Learning Rate | MAE  | RMSE |
|--------------|------|------|
| 0.05         | 16.16 | 21.22 |
| 0.01         | 15.79 | 20.98 |
| 0.005        | 15.93 | 19.89 |

Although RMSE improved at lower learning rates, XGBoost showed less stable behaviour near engine failure compared to Random Forest.

Increasing estimators at low learning rates led to overfitting.

---

## Target Engineering: RUL Capping

High RUL values introduce noise and are less operationally critical.

RUL was capped at 125 cycles:

RUL = min(RUL, 125)

Retraining Random Forest with capped RUL produced:

- MAE: 8.67  
- RMSE: 12.78  

Effects:

- Significant reduction in extreme errors.
- Tight clustering near failure.
- Conservative bias at high RUL (safer underprediction).

This demonstrated that target engineering had a larger impact than model choice.

---

## Deep Learning Models

Sequence modelling was implemented using raw 30-cycle windows.

### LSTM

- Dropout regularisation  
- Early stopping  

Results:

- MAE: 9.54  
- RMSE: 13.00  

---

### BiLSTM + Huber Loss

- Bidirectional LSTM  
- Huber loss for robustness  
- Reduced early stopping patience  

Results:

- MAE: 9.11  
- RMSE: 13.05  

Deep learning models performed competitively but did not outperform the capped Random Forest baseline.

---

## Model Comparison (Capped RUL)

| Model | MAE | RMSE |
|--------|------|------|
| Random Forest | 8.67 | 12.78 |
| LSTM | 9.54 | 13.00 |
| BiLSTM + Huber | 9.11 | 13.05 |

---

## Key Insights

- Classical feature engineering effectively captured degradation behaviour.
- RUL capping dramatically improved predictive stability.
- Deep learning did not provide substantial gains for FD001.
- Random Forest achieved the best balance between accuracy, stability near failure, conservative high-RUL behaviour, and simplicity.

---

## Conclusion

This project demonstrates that:

- Proper problem framing and target engineering significantly influence performance.
- Deep learning should be justified by dataset complexity, not assumed superior.
- For FD001, a capped Random Forest provides strong and reliable RUL prediction.

---

## Technologies Used

- Python  
- NumPy  
- pandas  
- scikit-learn  
- XGBoost  
- TensorFlow / Keras  
- Matplotlib  
