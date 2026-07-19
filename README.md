# Horse Racing Prediction System

This repository contains a system for collecting, processing, and predicting horse racing results based on historical data from Heppa (a Finnish horse racing website).

## Files

- **`all_heats_all_horses.py`**: Scrapes all race results from Heppa's website, extracting detailed information for each horse, including placing, odds, track conditions, and more. It saves the data to a CSV file.

- **`load_race_data_all_horses.py`**: A script to update the dataset by fetching new race results. It checks for duplicates to avoid reprocessing the same races.

- **`train_and_test_model.py`**: Trains a machine learning model (using LightGBM) to predict horse racing outcomes. The model uses features like temperature, track condition, horse and track history, and blood type to rank horses within each race.

- **`load_model_and_predict.py`**: Loads a pre-trained model and uses it to predict race outcomes for a given set of horses. It outputs predicted placing, odds, and other relevant metrics.

## Usage

1. Run `all_heats_all_horses.py` or `load_race_data_all_horses.py` to collect data.
2. Run `train_and_test_model.py` to train the model.
3. Run `load_model_and_predict.py` to make predictions on new races.

This system is designed for analyzing and predicting horse racing performance based on historical trends and conditions.