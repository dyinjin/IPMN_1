# NEED UPDATE

simple test:
```bash
--dataset all_d73 --param param_b
```

# Money Laundering Detection System

This repository contains a Python-based system for detecting money laundering activities in financial transaction data. The project is divided into two main components:

1. **ipmn_proflow**: A modular framework for transaction data processing and analysis
2. **Independent_demo**: Contains standalone demonstration scripts

## Project Structure

```
.
├── ipmn_proflow/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── division.py            # Data splitting strategies
│   ├── imports.py             # Centralized imports
│   ├── main.py                # Main execution script
│   ├── parameter_handler/
│   │   ├── __init__.py
│   │   ├── net_info_handler.py # Network feature extraction
│   │   └── time_handler.py     # Time-based feature extraction
│   └── unitdataloader.py      # Data loading utilities
├── Independent_demo/
│   ├── exemplary_main.py      # Standalone demo of ML model for money laundering detection
│   ├── network_view.py        # Network visualization for transaction relationships
│   └── separate.py            # Tool for separating dataset by month
```

## Installation Requirements

The system requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- networkx
- igraph
- imblearn (for SMOTE)

## ipmn_proflow Framework

The `ipmn_proflow` package provides a modular and configurable framework for money laundering detection.

### main.py

The primary execution script for the framework that:
1. Parses command-line arguments
2. Loads and processes data according to specified modes
3. Applies feature engineering
4. Splits data into training and testing sets
5. Trains and evaluates the model
6. Generates performance visualizations

Basic usage:
```bash
python main.py --dataset first_4 --param tdd_net_info_3 --division cut_7_train_3_test 
```

Advanced usage with parameters:
```bash
python main.py --dataset quick_test --param time_date_division --division random_7_train_3_test --balance default
```

Available arguments:
- `--dataset`: Data loading mode
  - `quick_test`: Uses predefined test dataset (default)
  - `all`: Uses the complete dataset
  - `first_2`: Uses first 2 months of data
  - `first_4`: Uses first 4 months of data
  - `IBM`: Uses IBM format data
  - `all_and_IBM`: Uses all data for training and IBM data for testing
  - `IBM_and_first_2`: Uses IBM data for training and first 2 months for testing

- `--param`: Feature engineering mode
  - `time_date_division`: Basic date/time feature extraction (default)
  - `tdd_net_info_1`: Time features + transaction count features
  - `tdd_net_info_2`: Time features + transaction counts + recent transaction partners
  - `tdd_net_info_3`: All features + network centrality measures

- `--division`: Train/test split strategy
  - `random_7_train_3_test`: Random 70/30 split (default)
  - `cut_7_train_3_test`: Sequential 70/30 split
  - `one_train_one_test`: Uses specific months for train and test
  - `rest_train_one_test`: Uses all but the last month for training

- `--balance`: Class balancing technique
  - `default`: No balancing (default)
  - `smote`: Uses SMOTE oversampling

### Other Framework Components

#### config.py

Defines the `Config` class to manage:
- Data paths
- Random seeds
- Input parameters
- Dataset modes
- Parameter handling modes
- Data division strategies
- Balance modes
- Model hyperparameters
- Command-line argument parsing

#### unitdataloader.py

Provides the `UnitDataLoader` class for standardized data loading:
- Loading data for specific year/month
- Loading complete datasets
- Loading data from IBM format
- Loading first N months of data

#### division.py

Implements custom data splitting strategies:
- `one_train_one_test`: Uses specific months for train and test
- `rest_train_one_test`: Uses all but the last month for training

#### parameter_handler/

Contains modules for feature engineering:

##### net_info_handler.py
- `net_info_tic`: Adds transaction count features
- `net_info_rtw`: Adds recent transaction partner features
- `net_info_3centrality`: Adds network centrality measures (degree, closeness, betweenness)

##### time_handler.py
- `date_apart`: Splits date/time into component features (year, month, day, hour, etc.)

## Independent_demo

### exemplary_main.py

A standalone script demonstrating the core money laundering detection workflow:

1. Loads transaction data from CSV
2. Performs feature engineering on date and transaction counts
3. Splits data into training and testing sets
4. Builds an XGBoost classifier with hyperparameter tuning
5. Evaluates model performance with ROC curves and confusion matrices

Usage:
```bash
python exemplary_main.py
```

### network_view.py

Visualizes transaction networks to identify potential money laundering patterns:

1. Loads transaction data
2. Filters for accounts involved in laundering activities
3. Creates a directed graph representation
4. Visualizes the network with color-coding for suspicious accounts

Usage:
```bash
python network_view.py
```

### separate.py

Utility script to split a large dataset into monthly CSV files:

1. Loads the master dataset
2. Separates data by month
3. Saves each month as an individual CSV file

Usage:
```bash
python separate.py
```

## Data Format

The system expects transaction data with the following columns:
- `Is_laundering`: Binary label indicating money laundering (target variable)
- `Date`: Transaction date
- `Time`: Transaction time
- `Sender_account`: Account initiating the transaction
- `Receiver_account`: Account receiving the transaction
- `Amount`: Transaction amount
- `Payment_currency`: Currency of payment
- `Received_currency`: Currency received
- `Payment_type`: Type of payment

## Model Performance

The system uses XGBoost for classification and evaluates performance using:
- ROC-AUC score
- Confusion matrix at specified TPR threshold (default 95%)
- Classification report (precision, recall, F1-score)


