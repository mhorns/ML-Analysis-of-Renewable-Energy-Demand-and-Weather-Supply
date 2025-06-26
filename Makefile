# Default target: run the sample pipeline
.PHONY: all
all: get_data_sample eda_sample naive_baseline_sample xgboost_sample LSTM_sample

# Run data acquisition sample script
.PHONY: get_data_sample
get_data_sample:
	@echo "Running get_data_sample.py..."
	python src/get_data_sample.py

# Run EDA sample script
.PHONY: eda_sample
eda_sample:
	@echo "Running regional_EDA_sample.py..."
	python src/regional_EDA_sample.py

# Run naive baseline predictions just for region sample
.PHONY: naive_baseline_sample
xgboost_sample:
	@echo "Running naive_baseline_eval.py (sample)..."
	python src/naive_baseline_eval.py

# Run XGBoost just for region sample
.PHONY: xgboost_sample
xgboost_sample:
	@echo "Running regional_XGBoost.py (sample)..."
	python src/regional_XGBoost.py

# Run LSTM just for region sample
.PHONY: LSTM_sample
xgboost_sample:
	@echo "Running regional_LSTM_GRU.py (sample)..."
	python src/regional_XGBoost.py

# Full pipeline (May take over 40 minutes to get data for each EIA and NASA)
.PHONY: full
full: get_data_full eda_full naive_baseline_full xgboost_full LSTM_full

.PHONY: get_data_full
get_data_full:
	@echo "Running full data download: this may take many minutes..."
	python src/get_data.py

.PHONY: eda_full
eda_full:
	@echo "Running full EDA..."
	python src/regional_EDA.py

.PHONY: naive_baseline_full
eda_full:
	@echo "Running naive_baseline_eval.py..."
	python src/naive_baseline_eval.py

.PHONY: xgboost_full
xgboost_full:
	@echo "Running full XGBoost model training: this may take over 40 minutes..."
	python src/regional_XGBoost.py

.PHONY: LSTM_full
xgboost_full:
	@echo "Running full LSTM model training: this may take over 40 minutes..."
	python src/regional_LSTM_GRU.py
