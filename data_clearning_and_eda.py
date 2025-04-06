#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Epileptic Seizure Dataset - Data Cleaning and Exploratory Data Analysis

Dataset Overview:
- Source: https://www.kaggle.com/datasets/chaditya95/epileptic-seizures-dataset
- Collected in a clinical setting
- Participants: 500
- Signal: 1-channel EEG signal
- Duration: 23.5 seconds per participant, 178 data points/second
- Label distribution: 80% Non-seizure (9200), 20% Seizure (2300)
"""

import numpy as np
import pandas as pd
import seaborn as sns  # Fixed typo: 'sn' -> 'sns'
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import argparse
import os


def load_data(file_path):
    """
    Load the dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)


def clean_data(df):
    """
    Clean the dataset by parsing row IDs, unifying versions, and converting labels.
    
    Args:
        df (pd.DataFrame): Original dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("Cleaning data...")
    
    # Parse Row ID
    print("1) Parsing row IDs...")
    # Split the first column into three new columns
    df[['second_no', 'version', 'pid']] = df.iloc[:, 0].str.split('.', expand=True)

    # Extract the numeric parts and replace X and V directly
    df['second_no'] = df['second_no'].str.extract('(\d+)').astype(int)  # Replace X with its numeric part
    df['version'] = df['version'].str.extract('(\d+)').astype(int)  # Replace V with its numeric part
    df['pid'] = df['pid'].fillna(0).astype(int)  # Convert the 'pid' column to integer for sorting

    # Some formatting
    # Rename the 'Unnamed: 0' column to 'id'
    df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    # Reorder the columns to move num, X, V between id and X1
    df = df[['id', 'pid', 'second_no', 'version', 'X1'] + 
            [col for col in df.columns if col not in ['id', 'pid', 'second_no', 'version', 'X1']]]
    
    # Unify version
    print("2) Unifying versions...")
    # Find the maximum existing pid
    max_pid = df['pid'].max()
    # Identify unique versions that are not 1
    versions_with_empty_pid = df[df['version'] != 1]['version'].unique()
    # Assign a unique pid to each participant and unify the version to 1
    for i, version in enumerate(versions_with_empty_pid):
        new_pid = max_pid + i + 1  # Start from max_pid + 1
        df.loc[(df['version'] == version), 'pid'] = new_pid
        df.loc[(df['version'] == version), 'version'] = 1

    # Convert labels to binary
    print("3) Converting labels to binary (0: Non-seizure, 1: Seizure)...")
    # Binary classification: 1: Seizure, 0: Non-seizure
    df['y'] = df['y'].apply(lambda x: 0 if x != 1 else x)
    
    print(f"After data cleaning:")
    print(f"There are {len(df['pid'].unique())} patients.")
    print(f"There are {len(df['second_no'].unique())} seconds per participant.")
    print(f"There are {len(df['version'].unique())} versions.")
    
    return df


def save_cleaned_data(df, output_path):
    """
    Save the cleaned dataframe to a CSV file.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        output_path (str): Path to save the cleaned data
    """
    print(f"Saving cleaned data to {output_path}...")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Data saved successfully.")


def check_label_consistency(df):
    """
    Check if all data for a single participant has the same label.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
    """
    print("\nChecking label consistency within participants...")
    # Group by 'pid' and check unique values of 'y'
    y_consistency = df.groupby('pid')['y'].nunique()

    # Identify pids where y does not stay the same
    inconsistent_pids = y_consistency[y_consistency > 1].index

    if len(inconsistent_pids) == 0:
        print("All participants have consistent labels across their data.")
    else:
        # Display inconsistent pids and their unique y values
        print("Found inconsistent labels for the following participants:")
        for pid in inconsistent_pids:
            unique_y_values = df[df['pid'] == pid]['y'].unique()
            print(f"PID: {pid} has inconsistent y values: {unique_y_values}")


def plot_mean_profiles(df):
    """
    Plot mean profiles of EEG signals for seizure and non-seizure classes.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
    """
    print("\nPlotting mean profiles...")
    
    # Get features and labels
    X = df.loc[:, 'X1':'X178'].values
    y = df['y'].values
    
    # Separate X based on y values
    X_class_0 = X[y == 0]
    X_class_1 = X[y == 1]

    # Compute the mean profile for each class
    mean_class_0 = np.mean(X_class_0, axis=0)
    mean_class_1 = np.mean(X_class_1, axis=0)

    # Plot the mean profiles
    plt.figure(figsize=(10, 6))
    plt.plot(mean_class_0, label='y=0 (Non-Seizure)', color='blue')
    plt.plot(mean_class_1, label='y=1 (Seizure)', color='red')
    plt.title('Mean Profiles of Classes y=0 and y=1')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/mean_profiles.png')
    plt.show()


def plot_feature_variability(df):
    """
    Plot standard deviation of features for seizure and non-seizure classes.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
    """
    print("Plotting feature variability...")
    
    # Get features and labels
    X = df.loc[:, 'X1':'X178'].values
    y = df['y'].values
    
    # Separate X based on y values
    X_class_0 = X[y == 0]
    X_class_1 = X[y == 1]

    # Compute standard deviation for each class
    std_class_0 = np.std(X_class_0, axis=0)
    std_class_1 = np.std(X_class_1, axis=0)

    # Plot the standard deviations
    plt.figure(figsize=(10, 6))
    plt.plot(std_class_0, label='y=0 (Non-Seizure)', color='blue')
    plt.plot(std_class_1, label='y=1 (Seizure)', color='red')
    plt.title('Feature Variability by Class')
    plt.xlabel('Feature Index')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/feature_variability.png')
    plt.show()


def plot_feature_heatmap(df):
    """
    Plot heatmap of mean feature values for seizure and non-seizure classes.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
    """
    print("Plotting feature heatmap...")
    
    # Get features and labels
    X = df.loc[:, 'X1':'X178'].values
    y = df['y'].values
    
    # Separate X based on y values
    X_class_0 = X[y == 0]
    X_class_1 = X[y == 1]

    # Compute mean feature values for each class
    mean_class_0 = np.mean(X_class_0, axis=0).reshape(1, -1)
    mean_class_1 = np.mean(X_class_1, axis=0).reshape(1, -1)

    # Combine into one array for visualization
    combined_means = np.vstack([mean_class_0, mean_class_1])

    # Plot heatmap
    plt.figure(figsize=(10, 4))
    sns.heatmap(combined_means, annot=False, cmap='coolwarm', xticklabels=False, yticklabels=['y=0', 'y=1'])
    plt.title('Heatmap of Mean Feature Values by Class')
    plt.xlabel('Feature Index')
    plt.ylabel('Class')
    plt.tight_layout()
    plt.savefig('plots/feature_heatmap.png')
    plt.show()


def plot_tsne_2d(df):
    """
    Plot 2D t-SNE visualization of the data.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
    """
    print("Generating 2D t-SNE visualization...")
    
    # Get data
    X = df.loc[:, 'X1':'X178'].values  # All feature columns
    y = df['y'].values  # Labels (non-seizure/seizure)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X)

    # Visualize the t-SNE features
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='coolwarm', style=y)
    plt.title('t-SNE of Seizure and Non-Seizure Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Label (0=Non-Seizure, 1=Seizure)')
    plt.tight_layout()
    plt.savefig('plots/tsne_2d.png')
    plt.show()


def plot_tsne_3d(df):
    """
    Plot 3D t-SNE visualization of the data.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
    """
    print("Generating 3D t-SNE visualization...")
    
    # Get data
    X = df.loc[:, 'X1':'X178'].values  # All feature columns
    y = df['y'].values  # Labels (non-seizure/seizure)

    # t-SNE with 3 components
    tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    X_tsne_3d = tsne.fit_transform(X)

    # Plot in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2],
        c=y, cmap='Spectral', marker='o', alpha=0.7
    )
    legend = ax.legend(
        *scatter.legend_elements(),
        title="Label (0=Non-Seizure, 1=Seizure)"
    )
    ax.add_artist(legend)
    ax.set_title('t-SNE of Seizure and Non-Seizure Data (3D)')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.tight_layout()
    plt.savefig('plots/tsne_3d.png')
    plt.show()


def perform_eda(df):
    """
    Perform exploratory data analysis on the cleaned data.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
    """
    print("\n## Exploratory Data Analysis (EDA)")
    
    # 1. Data imbalance
    print("\n1. Data imbalance:")
    label_counts = df['y'].value_counts()
    print(label_counts)
    print(f"Non-seizure (0): {label_counts[0]} ({label_counts[0]/len(df)*100:.1f}%)")
    print(f"Seizure (1): {label_counts[1]} ({label_counts[1]/len(df)*100:.1f}%)")
    
    # 2. Check label consistency
    check_label_consistency(df)
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # 3. Plot mean profiles
    plot_mean_profiles(df)
    
    # 4. Plot feature variability
    plot_feature_variability(df)
    
    # 5. Plot feature heatmap
    plot_feature_heatmap(df)
    
    # 6. Plot t-SNE visualizations
    plot_tsne_2d(df)
    plot_tsne_3d(df)


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Epileptic Seizure Dataset - Data Cleaning and EDA')
    parser.add_argument('--input', type=str, default='data/data.csv',
                        help='Path to the input CSV file')
    parser.add_argument('--output', type=str, default='data/data_cleaned.csv',
                        help='Path to save the cleaned CSV file')
    parser.add_argument('--skip-eda', action='store_true',
                        help='Skip the EDA part and only perform data cleaning')
    return parser.parse_args()


def main():
    """
    Main function to run the data cleaning and EDA pipeline.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load data
    df = load_data(args.input)
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Save cleaned data
    save_cleaned_data(df_cleaned, args.output)
    
    # Perform EDA if not skipped
    if not args.skip_eda:
        perform_eda(df_cleaned)
    
    print("\nData cleaning and EDA completed successfully!")


if __name__ == "__main__":
    main()