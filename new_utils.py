"""
Thomas Schuff
Professor MacIsaac
CPSC 222 - Data Science
Quantified Self Project - "LaLaLand"

Utility functions for analyzing my daily Apple Music listening history. This file contains all the project logic code and is called from lalaland.ipynb.
Functions have been organized into the following sections:
    - Data Loading --> Load "apple_music_play_activity.csv" and "weekday_table.csv"
    - Data Cleaning --> Remove irrelevant columns, standardize timestamps, and collapse duplicate play events into listening "sessions"
    - Data Aggregation --> Calculate daily listening session counts
    - Merging Tables --> Join the activity data with the weekday lookup table
    - Adding Features --> Add month, rolling average, and weekend flag columns
    - Data Visualization --> Plot daily trends, box plots, bar charts, and the rolling average plot
    - Hypothesis Testing --> Two-sample t-Test (weekday vs. weekend) and one-way ANOVAs (by month and by day of week)
    - Classification --> Train and evaluate kNN and Decision Tree classifiers to predict whether a day is a weekday or weekend
"""

# --- IMPORTS ---
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree


# --- DATA LOADING ---
def load_activity_data(filename):
    """
    Load a CSV file into a Pandas DataFrame.
    """

    return pd.read_csv(filename, low_memory=False)


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
    df = df.groupby(["Song Name", "Date"], group_keys=False).apply(calc_song_session)

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
    Add True/False weekend indicator (as an integer 0/1) for the classification.
    """

    df["Is Weekend"] = df["Day of Week"].isin(["Saturday", "Sunday"]).astype(int)
    return df


# --- DATA VISUALIZATION ---
def plot_daily_counts(df):
    """
    Plot the daily listening counts
    """
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Listen Count"])
    plt.title("Daily Listening Sessions Over Time")
    plt.xlabel("Date")
    plt.ylabel("Listen Count")
    plt.ylim(
        bottom=0,
        top=300,
    )
    plt.xticks(rotation=45)
    plt.show()


def boxplot_wkday_wkend(df):
    """
    Generate a boxplot comparing the distributions of weekday and weekend listening counts.
    """

    weekday = df[df["Is Weekend"] == 0]["Listen Count"]
    weekend = df[df["Is Weekend"] == 1]["Listen Count"]

    wkday_median = weekday.median()
    wknd_median = weekend.median()

    plt.figure(figsize=(6, 5))
    plt.boxplot([weekday, weekend], tick_labels=["Weekday", "Weekend"])
    plt.annotate(
        f"Median: {wkday_median:.0f}",
        xy=(1, wkday_median),
        xytext=(1.1, wkday_median),
        fontsize=9,
        va="center",
    )
    plt.annotate(
        f"Median: {wknd_median:.0f}",
        xy=(2, wknd_median),
        xytext=(2.1, wknd_median),
        fontsize=9,
        va="center",
    )
    plt.title("Weekday vs Weekend Listening")
    plt.xlabel("Day Type")
    plt.ylabel("Listen Count")
    plt.ylim(bottom=0, top=300)
    plt.show()


def plot_listening_by_day_of_week(df):
    """
    Bar chart showing average listening counts for each day of the week.
    """
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    day_avg = df.groupby("Day of Week")["Listen Count"].mean().reindex(days)

    plt.figure(figsize=(8, 5), layout="constrained")
    plt.bar(x=day_avg.index, height=day_avg.values)
    plt.title("Average Listening Count by Day of the Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Mean Listen Count (# Songs Played)")
    plt.xticks(rotation=45)
    plt.show()


def monthly_trend(df):
    """
    Generate a bar plot comparing the average listening counts for each month
    """

    monthly_avg = df.groupby("Month")["Listen Count"].mean()
    plt.figure(figsize=(8, 5))
    plt.bar(monthly_avg.index, monthly_avg.values)
    plt.title("Average Listening Count by Month")
    plt.xlabel("Month")
    plt.ylabel("Mean Listen Count")
    plt.xticks(monthly_avg.index)
    plt.show()


def rolling_average_trend(df):
    """
    Generate a line plot with the rolling average on top of the daily listening counts
    """

    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Listen Count"], alpha=0.4, label="Daily Count")
    plt.plot(
        df["Date"],
        df["Rolling Average"],
        color="red",
        label="7-Day Rolling Avg",
    )
    plt.title("Daily Listening Sessions with 7-Day Rolling Average")
    plt.xlabel("Date")
    plt.ylabel("Listen Count")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


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


def anova_day_of_week(df):
    """
    One-way ANOVA comparing listening counts across the days of the week.
    Returns the f-statistic and p-value.
    """

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    groups = [df[df["Day of Week"] == day]["Listen Count"] for day in days]
    return stats.f_oneway(*groups)


def anova_monthly(df):
    """
    One-way ANOVA comparing listening counts across months.
    Returns the f-statistic and p-value.
    """

    groups = [group["Listen Count"].values for _, group in df.groupby("Month")]
    return stats.f_oneway(*groups)


# --- CLASSIFICATION (kNN and Decision Tree) ---
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


def train_decision_tree(X, y, max_depth=5, test_size=0.22, random_state=42):
    """
    Train a Decision Tree classifier to predict weekday vs. weekend.
    Returns the model, X_test, y_test, y_pred, and accuracy.
    Uses the same train, test, split parameters as the kNN so comparisons can be made between the two.
    """

    # Train, test, split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Decision Tree
    tree_clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree_clf.fit(X_train_scaled, y_train)

    # Predict and accuracy
    y_pred = tree_clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    return tree_clf, X_test, y_test, y_pred, acc


def plot_decision_tree(tree_clf, feature_cols):
    """
    Visualize the trained Decision Tree using MatPlotLib
    Shows the feature splits, entropy, and class predictions at each node.
    """
    fig, ax = plt.subplots(layout="constrained", figsize=(14, 6))
    plot_tree(
        tree_clf,
        feature_names=feature_cols,
        class_names=["Weekday", "Weekend"],
        filled=True,
        rounded=True,
        ax=ax,
    )
    ax.set_title(
        "Decision Tree: Predicting Weekday vs. Weekend from Listening Behavior"
    )
    plt.show()


def convert_label(value):
    """
    Converts the "Is Weekend" values from 0/1 to "Weekday" or "Weekend"
    """

    if value == 0:
        return "Weekday"
    else:
        return "Weekend"


def clf_report(y_test, y_pred, acc, title="Classification Results"):
    print(f"{title}")
    print(f"Accuracy: {acc:.4f}\n")

    # Confusion Matrix (text display)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm, "\n")

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Weekday", "Weekend"]))

    # Confusion Matrix (visual display)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Weekday", "Weekend"]
    )
    fig, ax = plt.subplots(layout="constrained", figsize=(5, 4))
    disp.plot(ax=ax, colorbar=True)
    ax.set_title(f"{title} - Confusion Matrix")
    plt.show()

    # Show first few predictions
    results_df = pd.DataFrame(
        {
            "Actual": [convert_label(act_val) for act_val in y_test],
            "Predicted": [convert_label(pred_val) for pred_val in y_pred],
        }
    )

    print("\nSample Predictions:")
    print(results_df.head(50))
