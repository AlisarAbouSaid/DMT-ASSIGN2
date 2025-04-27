import pandas as pd
import pickle
import os
import gc
import numpy as np
import re

def load_data(filename):
    # Loads CSV into pandas DataFrame
    return pd.read_csv(filename)

def add_date_features(df):
    # Make sure 'srch_query_affinity_score' and 'position' exist
    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
        df['year'] = df['date_time'].dt.year
        df['month'] = df['date_time'].dt.month
        df['day'] = df['date_time'].dt.day
        df['hour'] = df['date_time'].dt.hour
    return df

def normalize_features(df):
    # Normalizes srch_query_affinity_score and position
    if 'srch_query_affinity_score' in df.columns:
        df['srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(df['srch_query_affinity_score'].mean())
        df['srch_query_affinity_score'] = (df['srch_query_affinity_score'] - df['srch_query_affinity_score'].mean()) / df['srch_query_affinity_score'].std()
    if 'position' in df.columns:
        df['position'] = (df['position'] - df['position'].mean()) / df['position'].std()
    return df

def fill_missing(df):
    # Fill missing numerical values with mean
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    return df

def preprocess_training_data(df, kind="train"):
    df = add_date_features(df)
    df = normalize_features(df)
    df = fill_missing(df)

    labels = None
    if kind == "train":
        # Suppose 'click_bool' or 'booking_bool' are your labels
        if 'click_bool' in df.columns and 'booking_bool' in df.columns:
            labels = df[['click_bool', 'booking_bool']].copy()
            df = df.drop(columns=['click_bool', 'booking_bool'])

    return df, labels

# Main auto-run block
if __name__ == "__main__":
    print("Starting data preprocessing...")

    input_file = "training_set_VU_DM.csv"  # Change this if needed
    output_file = "train_preprocessed.pkl"

    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found! Please check the path.")
        exit(1)

    # Load original data
    orig_data = load_data(input_file)

    # Preprocess
    processed_data, labels = preprocess_training_data(orig_data, kind="train")

    # Save results
    with open(output_file, "wb") as f:
        pickle.dump((processed_data, labels), f)

    print(f"Preprocessing done! Data saved to '{output_file}'.")
    import pickle

    # Example: open a file called "model.pkl"
    with open('train_preprocessed.pkl', 'rb') as file:
        data = pickle.load(file)

    # Now 'data' contains whatever was saved inside the .pkl file
    print(data)

    # Garbage collection
    del orig_data
    gc.collect()
