# Default target: run the sample pipeline
.PHONY: all
all: get_data_sample eda_sample

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

# Full pipeline (May take over 40 minutes to get data for each EIA and NASA)
.PHONY: full
full: get_data_full eda_full

.PHONY: get_data_full
get_data_full:
	@echo "Running full data download: this may take many minutes..."
	python src/get_data.py

.PHONY: eda_full
eda_full:
	@echo "Running full EDA..."
	python src/regional_EDA.py

