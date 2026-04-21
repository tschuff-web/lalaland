"""
Thomas Schuff
Professor MacIsaac
CPSC 222 - Data Science
Quantified Self Project - "LaLaLand"

Utility functions for analyzing my daily Apple Music listening history.
    - Loading data
    - Cleaning timestamps
    - Collapsing duplicate song events into "listening sessions"
    - Computing daily listening counts
"""

# --- IMPORTS ---
import pandas as pd
import numpy as np
import csv
import matplotlib as mpl
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import json


# --- DATA LOADING ---
def load_activity_data(filename):
    """
    Load a CSV file into a Pandas DataFrame.
    """
    return pd.read_csv(filename)  # type: ignore


def load_weekday_table(filename):
    """
    Load a table mapping dates to weekday/weekend labels
    """
    return pd.read_csv(filename)


# --- DATA CLEANING ---
def clean_activity_data(df):
    """
    Clean the raw Apple Music activity data by:
        - removing irrelevant columns
        - converting timestamps
        - collapsing duplicate song events into "listening sessions"
        - adding a "date" column for daily aggregation
    """

    # Load the list of columns to remove
    with open(
        "lalaland/dataset files/columns_to_remove.csv",
        mode="r",
        newline="",
        encoding="utf-8",
    ) as col_file:
        reader = csv.reader(col_file)
        cols_to_remove = next(reader)

    # Strip quotes and whitespace
    cols_to_remove = [col.strip().strip("'") for col in cols_to_remove]

    # Remove irrelevant columns
    df = df.drop(columns=cols_to_remove)

    # Rename columns to reduce typing lol
    df.rename(columns={"Event Received Timestamp": "Event Timestamp"}, inplace=True)

    # Convert timestamps to DateTime objects
    df["Event Timestamp"] = pd.to_datetime(
        df["Event Timestamp"], format="mixed", utc=True
    )

    # Add a date column
    df["Date"] = df["Event Timestamp"].dt.date

    # Collapse duplicate song events (Apple records event every time you pause, play, etc. so lots of duplicates)
    df = df.groupby(["Song Name", "Date"], group_keys=False).apply(calc_song_session)  # type: ignore

    # Keep only the first event of each session
    df = df[df["New Session"]]

    return df


def calc_song_session(group):
    """
    Collapse multiple Apple Music play events into a single "session".
    Apple logs events for play, pause, resume, etc. which creates a lot of duplicate events. This function:
        - Sorts events by timestamp
        - Calculates a time difference between consecutive events
        - Marks it as a new session if the difference is greater than 5 minutes
    """
    group = group.sort_values("Event Timestamp")
    group["Time Difference"] = group["Event Timestamp"].diff()

    group["New Session"] = (group["Time Difference"] > pd.Timedelta(minutes=5)) | (
        group["Time Difference"].isna()
    )

    return group


# --- DATA AGGREGATION ---
def calc_daily_counts(df):
    """
    Calculate the number of unique listening sessions per day
    Each "session" represents one "listen" of a song
    """
    daily_counts = df.groupby("Date").size().reset_index(name="Listen Count")
    return daily_counts


# --- MERGING TABLES ---
def merge_with_weekday(activity_df, weekday_df):
    """
    Combining my two tables! Merges the daily listening data with the weekday/weekend table.
    """

    # Convert to DateTime objects and extract the dates for both tables
    weekday_df["Date"] = pd.to_datetime(weekday_df["Date"]).dt.date
    activity_df["Date"] = pd.to_datetime(activity_df["Date"]).dt.date

    merged_df = activity_df.merge(weekday_df, on="Date", how="left")
    return merged_df


# --- ADDING FEATURES ---
def add_rolling_average(df, window=7):
    """
    Add a rolling average of daily listening counts.
    """
    df = df.sort_values("Date")
    df["Rolling Average"] = df["Listen Count"].rolling(window=window).mean()
    return df


def add_month(df):
    """
    Add a month column for analyzing seasonal trends.
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    return df


def add_weekend_flag(df):
    """
    Add True/False weekend indicator for the classification.
    """
    df["Is Weekend"] = df["Day of Week"].isin(["Saturday", "Sunday"]).astype(int)
    return df


# --- HYPOTHESIS TESTING ---
def t_test_weekday_vs_weekend(df):
    """
    Two-sample t-test comparing weekday vs. weekend listening counts
    Returns the t-statistic and p-value
    """
    weekday = df[
        df["Day of Week"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    ]["Listen Count"]

    weekend = df[df["Day of Week"].isin(["Saturday", "Sunday"])]["Listen Count"]

    return stats.ttest_ind(weekday, weekend, equal_var=False)


def anova_monthly(df):
    """
    One-way ANOVA comparing listening counts across months.
    Returns the f-statistic and p-value
    """

    groups = [group["Listen Count"].values for _, group in df.groupby("Month")]
    return stats.f_oneway(*groups)


# --- CLASSIFICATION (kNN) ---
def prep_classification(df, feature_cols, class_col):
    """
    Prepare X and y for kNN and Decision Tree classification.
    """
    X = df[feature_cols]
    y = df[class_col]
    return X, y


def train_knn_classifier(X, y, k=5, test_size=0.2, random_state=42):
    """
    Train a kNN classifier to predict weekday vs. weekend.
    Returns the model, X_test, y_test, y_pred, and accuracy
    """

    # Train, test, split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train kNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    # Predict and accuracy
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    return knn, X_test, y_test, y_pred, acc


def convert_label(value):
    """
    Converts the "Is Weekend" values from 0/1 to "Weekday" or "Weekend"
    """
    if value == 0:
        return "Weekday"
    else:
        return "Weekend"


def clf_report(y_test, y_pred, acc):
    print("kNN Classification Results")
    print(f"Accuracy: {acc:.4f}\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm, "\n")

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Weekday", "Weekend"]))

    # Show first few predictions
    results_df = pd.DataFrame(
        {
            "Actual": [convert_label(act_val) for act_val in y_test],
            "Predicted": [convert_label(pred_val) for pred_val in y_pred],
        }
    )

    print("\nSample Predictions:")
    print(results_df.head(50))
