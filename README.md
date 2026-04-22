# _La La Land_ - Analyzing my Apple Music Listenign Behavior

## Quantified Self Project - CPSC 222 (Spring 2026)

### Thomas Schuff

---

## Overview

LaLaLand is a Quantified Self project that analyzes my daily Apple Music
listening history over the last year(ish). The goal is to understand how my
listening behavior changes across time, test hypotheses about weekday/weekend
and monthly differences, and build classifiers that predict whether a given day
is a weekday or weekend solely on listening patterns.

This project includes the following:

- Data cleaning and preprocessing
- Merging multiple tables
- Exploratory data analysis (EDA)
- Statistical hypothesis testing
- Machine learning classification (kNN)

---

### Key Components

- **lalaland.ipynb**  
  The main narrative report containing the introduction, data preparation, EDA,
  hypothesis testing, and classification results. Code cells are interleaved and
  mostly call functions from the `new_utils.py` file.

- **new_utils.py**  
  A utility module containing all the data loading, cleaning, aggregation,
  visualization, hypothesis testing, and classification functions (kNN and
  Decision Tree).

- **dataset files/**  
  Contains the Apple Music export, the weekday/weekend table, and a list of
  columns to remove during cleaning.

---

## How to Run the Project

### Dependencies

This project uses Python 3.10+ and the following libraries:

- pandas
- numpy
- matplotlib
- scipy
- scikit‑learn

Importantly, remember to set your working directory to: %cd
/path/to/quantified-self-project

Then, within the `lalaland.ipynb` Jupyter Notebook, execute the Notebook cells
in order with `Run All`. The notebook will then:

1. Load the Apple Music dataset
2. Clean and preprocess the data
3. Merge with the weekday table
4. Perform EDA and hypothesis testing
5. Train and evaluate the kNN classifier

### Data Sources

1. Apple Music Play Activity - Exported from Apple’s privacy portal
   ([https://privacy.apple.com](https://privacy.apple.com)).
   - Contains timestamped events for every play, pause, and resume action.
2. Weekday Table
   - Generated manually based on calendar dates. No third‑party datasets were
     used.

### Notes on Ethics and Privacy

- This project uses personal listening data.
- All analysis is performed locally, and no data is shared or uploaded
  externally.
- The project demonstrates responsible handling of personal digital traces.
