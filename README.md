# _La La Land_ - Analyzing my Apple Music Listening Behavior

## Quantified Self Project - CPSC 222 (Spring 2026)

### Thomas Schuff

---

## Overview

LaLaLand is a Quantified Self project that analyzes my daily Apple Music
listening history over the last year-ish (June 2025 to April 2026). The goal is
to better understand how my listening behavior changes across time, test
hypotheses about weekday/weekend and monthly differences, and build classifiers
that predict whether a given day is a weekday or weekend based on listening
patterns.

This project includes the following:

- Data cleaning and preprocessing
- Merging multiple tables from different data sources
- Exploratory data analysis (EDA) with five visualizations
- Statistical hypothesis testing (t-Test and two ANOVAS)
- Machine learning classification (kNN and Decision Tree)

---

### Key Components

- `lalaland.ipynb` The main narrative report containing the introduction, data
  preparation, EDA, hypothesis testing, and classification results. Code cells
  are short and mostly call functions from `new_utils.py`, with surrounding
  markdown cells describing inputs, outputs, and insights.

- `new_utils.py` A utility file with all the data loading, cleaning,
  aggregation, visualization, hypothesis testing, and classification functions
  (kNN and Decision Tree). All project logic lives here, and the notebook calls
  these functions directly

- `dataset files` Contains three input files
  - `Apple Music Play Activity.csv` - raw export from Apple
  - `weekday_table.csv` - manually created lookup table with calendar date and
    day of week
  - `columns_to_remove.csv` - list of irrelevant columns that are dropped during
    cleaning
- `output files` Contains the three output CSV files from various stages of data
  cleaning.

---

## How to Run the Project

### Dependencies

This project uses Python 3.10+ and the following libraries:

- pandas
- numpy
- matplotlib
- scipy
- scikit‑learn

### Running the Notebook

Set your working directory to: %cd /path/to/quantified-self-project

Run all cells in order with `Run All`. The notebook will:

1. Load the Apple Music dataset and weekday table
2. Clean and preprocess the data
3. Merge the two tables on the `Date` column
4. dd features (month, rolling average, weekend flag)
5. Perform EDA and hypothesis testing
6. Train and evaluate the kNN and Decision Tree classifiers

## Dataset

### Data Sources

1. **Apple Music Play Activity** - Exported from Apple’s privacy portal
   ([https://privacy.apple.com](https://privacy.apple.com)).
   - Contains timestamped events for every play, pause, and resume action.
   - Raw table: **79,403** rows; cleaned to **38,843** unique listening
     "sessions"
2. **Weekday Table** - Generated manually in Excel from calendar dates.
   - Maps each date to its corresponding day of the week
   - **340** rows, one per date in the dataset.

### Attributes (after cleaning and merging)

| Attribute         | Description                           | Scale    |
| ----------------- | ------------------------------------- | -------- |
| `Date`            | Calendar Date                         | Interval |
| `Listen Count`    | Unique listening sessions per day     | Ratio    |
| `Day of Week`     | Monday through Sunday                 | Nominal  |
| `Is Weekend`      | Class label: 0 = Weekday, 1 = Weekend | Binary   |
| `Month`           | Numeric month (1-12)                  | Ordinal  |
| `Rolling Average` | 7-day smoothed listening trend        | Ratio    |
