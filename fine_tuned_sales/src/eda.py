# src/eda.py
# EDA stands for Exploratory Data Analysis. 
# It’s the very first step you take when you get a new dataset—before you build any models—to understand what you’re working with.

import pandas as pd
import matplotlib.pyplot as plt

def load_data(path="data/sales_transcripts.csv"):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} transcripts")
    print(df.head())
    return df

def plot_label_distribution(df):
    counts = df["label"].value_counts()
    counts.plot.bar(title="Won vs. Lost Counts")
    plt.ylabel("Number of Calls")
    plt.show()

def plot_transcript_length(df):
    df["length"] = df["transcript"].str.split().apply(len)
    df["length"].hist(bins=10)
    plt.title("Transcript Length Distribution (in words)")
    plt.xlabel("Words per Transcript")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    df = load_data()
    plot_label_distribution(df)
    plot_transcript_length(df)
