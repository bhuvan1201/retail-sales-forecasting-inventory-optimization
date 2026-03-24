# Retail Sales Forecasting & Inventory Optimization Engine

## Overview
This project builds an end-to-end retail demand forecasting and inventory optimization workflow using the Walmart M5 dataset. The goal is to forecast product demand, compare multiple forecasting models, and translate forecasts into actionable inventory decisions such as reorder points and safety stock.

## Business Problem
Retailers need accurate demand forecasts to avoid:
- Stockouts that cause lost sales
- Overstock that increases holding costs
- Poor replenishment decisions across stores and categories

This project solves that by:
1. Preparing raw retail sales, calendar, and pricing data
2. Engineering time-series and business features
3. Comparing ARIMA, Prophet, and XGBoost forecasting models
4. Simulating inventory decisions based on forecasted demand
5. Preparing outputs for dashboarding and reporting

## Dataset
The project uses the Walmart M5 forecasting dataset:

- `calendar.csv`
- `sales_train_validation.csv`
- `sales_train_evaluation.csv`
- `sell_prices.csv`
- `sample_submission.csv`

## Planned Workflow
1. Raw data ingestion
2. Sales reshaping from wide to long format
3. Merge with calendar and price data
4. Feature engineering
5. Train/validation split
6. Forecasting model comparison
7. Inventory simulation and KPI generation
8. Dashboard-ready reporting layer

## Repository Structure
```text
data/raw              # Original dataset files
data/interim          # Intermediate parquet files
data/processed        # Final train/validation/forecast outputs
notebooks/            # EDA and experimentation
src/                  # Core project code
configs/              # Config files
outputs/              # Figures, reports, tables
docs/                 # Methodology and assumptions