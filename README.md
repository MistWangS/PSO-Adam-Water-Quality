# Hybrid Intelligent Optimization for Water Quality Monitoring

This repository contains the core codebase and sample data for the paper: **"An Intelligent Expert System for Inland Water Quality Monitoring via Multi-source Satellite Fusion and Hybrid PSO-Adam Optimization Strategy."**

## 1. Overview
This intelligent expert system is designed to perform high-precision inversion of Total Nitrogen (TN) and Total Phosphorus (TP) in complex inland water bodies. It integrates an indirect inversion strategy with a hybrid mathematical optimization framework. Specifically, it employs a stochastic global search mechanism to avoid non-convex local optima, subsequently coupled with Adam's adaptive gradient descent for fine-tuning machine learning networks.

## 2. Repository Structure
* `main.py`: The core execution script containing data preprocessing, the stochastic global search algorithm, neural network architecture, and evaluation metrics.
* `data-sd-fan+chla.xlsx`: A sample dataset containing remote sensing derived features (SD, temp, Chl-a) and ground-truth Total Nitrogen (TN) measurements.

## 3. Dependencies & Requirements
To execute this framework, the following Python libraries are required:
* Python 3.8+
* PyTorch (`torch`)
* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`
* `openpyxl` (for exporting results)

## 4. Usage Instructions
1. Clone this repository or download the source files.
2. Ensure that the sample data file (`data-sd-fan+chla.xlsx`) is placed in the same root directory as the main script.
3. Run the main optimization script:
   ```bash
   python main.py