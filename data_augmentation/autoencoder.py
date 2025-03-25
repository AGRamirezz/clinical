# -*- coding: utf-8 -*-
"""
Autoencoder for EEG Signal Augmentation

This module implements an autoencoder architecture for augmenting EEG signals
to improve seizure detection. The autoencoder uses an Inception Nucleus design
to capture multi-scale temporal patterns in EEG data.

Usage:
    1. Train the autoencoder on EEG data
    2. Use the trained autoencoder to generate augmented signals
    3. Evaluate the impact on seizure detection performance

The architecture is based on: https://ieeexplore.ieee.org/document/9054725
"""

import os
import time
import logging
from typing import Tuple, List, Dict, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, cohen_kappa_score
)

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Input, Conv1D, Conv2D, MaxPooling2D, BatchNormalization, 
    Reshape, UpSampling2D, Conv2DTranspose, Conv1DTranspose, 
    Dropout, Dense, Flatten
)


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure logging for the application.
    
    Args:
        log_file: Optional path to log file
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('autoencoder')
    logger.setLevel(level)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Initialize logger
logger = setup_logging(log_file="autoencoder_run.log")


def setup_device() -> str:
    """Configure and return the appropriate device for computation."""
    gpus = tf.config.list_physical_devices('GPU')
    device = '/device:GPU:0' if gpus else '/device:CPU:0'
    logger.info(f"Using device: {device}")
    return device


def create_checkpoint_callback(checkpoint_dir: str = "checkpoints", 
                              model_name: str = "autoencoder") -> ModelCheckpoint:
    """Create a callback for model checkpointing.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        model_name: Base name for the model files
        
    Returns:
        ModelCheckpoint callback
    """
    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint path with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(
        checkpoint_dir, 
        f"{model_name}_{timestamp}" + "_{epoch:02d}-{val_loss:.4f}.h5"
    )
    
    # Create callback to save best model
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    return checkpoint_callback


def load_latest_checkpoint(checkpoint_dir: str = "checkpoints", 
                          model_name: str = "autoencoder") -> Optional[Model]:
    """Load the latest checkpoint for a model.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model_name: Base name of the model files
        
    Returns:
        Loaded model or None if no checkpoint found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find all checkpoints for this model
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(model_name)]
    
    if not checkpoints:
        return None
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
    
    # Load the model
    try:
        logger.info(f"Loading checkpoint: {latest_checkpoint}")
        return load_model(latest_checkpoint)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


def inception_nucleus_encoder(input_layer: tf.Tensor, 
                             filters: List[int], 
                             dropout_rate: float = 0) -> tf.Tensor:
    """
    Implement the Inception Nucleus for the encoder.
    
    Args:
        input_layer: Input tensor
        filters: List of kernel sizes for parallel convolutions
        dropout_rate: Dropout rate to apply
        
    Returns:
        Output tensor after multi-scale convolution
    """
    branches = [
        Dropout(dropout_rate)(
            BatchNormalization()(
                Conv1D(filters=32, kernel_size=f, padding='same', activation='relu')(input_layer)
            )
        ) for f in filters
    ]
    return tf.keras.layers.concatenate(branches, axis=-1)


def inception_nucleus_transpose(input_layer: tf.Tensor, 
                               filters: List[int]) -> tf.Tensor:
    """
    Implement the reverse Inception Nucleus for the decoder.
    
    Args:
        input_layer: Input tensor
        filters: List of kernel sizes for parallel transposed convolutions
        
    Returns:
        Output tensor after multi-scale transposed convolution
    """
    branches = [
        Conv1DTranspose(filters=32, kernel_size=f, padding='same', activation='relu')(input_layer)
        for f in filters
    ]
    return tf.keras.layers.concatenate(branches, axis=-1)


def create_eeg_autoencoder(one_d_dropout_rate: float = 0, 
                          two_d_dropout_rate: float = 0) -> Model:
    """
    Create an autoencoder model for EEG signal processing.
    
    Args:
        one_d_dropout_rate: Dropout rate for 1D convolutional layers
        two_d_dropout_rate: Dropout rate for 2D convolutional layers
        
    Returns:
        Compiled autoencoder model
    """
    # Input layer - expects flattened EEG signals of length 178
    inputs = Input(shape=(178,))
    x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)  # Add channel dimension

    # Encoder: Multi-scale feature extraction with Inception Nucleus
    x = inception_nucleus_encoder(x, filters=[4, 8, 12], dropout_rate=one_d_dropout_rate)

    # Reshape for 2D convolutions
    x = Reshape((x.shape[1], x.shape[2], 1))(x)

    # 2D convolutional layer with regularization
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(two_d_dropout_rate)(x)
    encoded = MaxPooling2D((2, 2))(x)  # Compressed representation

    # Decoder: Reverse the encoder process
    x = UpSampling2D((2, 2))(encoded)
    x = Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Reshape back to 1D format
    x = Reshape((178, 96 * 64))(x)  # Flatten the feature maps
    x = Conv1D(96, kernel_size=1, activation='relu')(x)  # Reduce dimensionality

    # Reverse the multi-scale convolutions
    x = inception_nucleus_transpose(x, filters=[4, 8, 12])

    # Final reconstruction layer
    outputs = Conv1D(1, kernel_size=1, padding='same', activation='sigmoid')(x)

    # Create and compile the model
    autoencoder = Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


def load_train_hard_test_split():
    """
    Load and preprocess the dataset with a predefined train/test split.
    
    Returns:
        Tuple containing:
        - X_train: Raw training features
        - X_train_scaled: Normalized training features
        - X_test: Raw test features
        - X_test_scaled: Normalized test features
        - y_train: Training labels
        - y_test: Test labels
        - scaler: Fitted MinMaxScaler
    """
    train_data = pd.read_csv('../data/hard_test_cases/train.csv', index_col=0)
    test_data = pd.read_csv('../data/hard_test_cases/test.csv', index_col=0)

    X_train = train_data.iloc[:, :-1].values  # X1-X178
    X_test = test_data.iloc[:, :-1].values    # X1-X178

    # Normalize the data using min-max scaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape the data for the model
    X_train_scaled = X_train_scaled.reshape(-1, 178, 1)
    X_test_scaled = X_test_scaled.reshape(-1, 178, 1)

    # Extract labels
    y_train = train_data['y'].values
    y_test = test_data['y'].values

    return X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test, scaler


def train_autoencoder(device: str, 
                     X_train: np.ndarray, 
                     X_test: np.ndarray, 
                     epochs: int = 10, 
                     **kwargs) -> Model:
    """
    Train an autoencoder model for EEG signal reconstruction.
    
    Args:
        device: Device to use for training ('/device:GPU:0' or '/device:CPU:0')
        X_train: Training data
        X_test: Validation data
        epochs: Number of training epochs
        **kwargs: Additional parameters for autoencoder creation
        
    Returns:
        Trained autoencoder model
    """
    with tf.device(device):
        # Try to load existing checkpoint
        model_name = "autoencoder_dropout" if kwargs.get('one_d_dropout_rate', 0) > 0 else "autoencoder"
        loaded_model = load_latest_checkpoint(model_name=model_name)
        
        if loaded_model:
            logger.info(f"Resuming training from checkpoint")
            autoencoder = loaded_model
        else:
            logger.info(f"Creating new autoencoder model with dropout rates: 1D={kwargs.get('one_d_dropout_rate', 0)}, 2D={kwargs.get('two_d_dropout_rate', 0)}")
            autoencoder = create_eeg_autoencoder(**kwargs)
        
        # Create checkpoint callback
        checkpoint_callback = create_checkpoint_callback(model_name=model_name)
        
        # Train the model
        logger.info(f"Training autoencoder for {epochs} epochs")
        autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, X_test),
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True),
                checkpoint_callback
            ],
            verbose=1
        )
        
        # Evaluate the model
        val_loss = autoencoder.evaluate(X_test, X_test, verbose=0)
        logger.info(f"Validation loss: {val_loss:.4f}")
        
        return autoencoder


def visualize_signals(original_signal: np.ndarray, 
                     augmented_signal: np.ndarray, 
                     title: str) -> None:
    """
    Visualize original and augmented signals.
    
    Args:
        original_signal: Original time-series signal
        augmented_signal: Augmented time-series signal
        title: Plot title
    """
    plt.figure(figsize=(10, 4))
    plt.plot(original_signal, label="Original Signal", color="blue", alpha=0.7)
    plt.plot(augmented_signal, label="Augmented Signal", color="red", alpha=0.7)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


def generate_augmented_signals(autoencoder: Model, 
                              X: np.ndarray, 
                              num_iterations: int = 5) -> List[np.ndarray]:
    """
    Generate augmented signals by iteratively passing through the autoencoder.
    
    Args:
        autoencoder: Trained autoencoder model
        X: Input signals to augment
        num_iterations: Number of iterations for augmentation
        
    Returns:
        List of augmented signals
    """
    logger.info(f"Generating augmented signals with {num_iterations} iterations")
    augmented_signals = []
    current_signals = X.copy()
    
    for i in range(num_iterations):
        # Reshape for prediction if needed
        if len(current_signals.shape) == 3:  # (samples, timesteps, channels)
            input_signals = current_signals.reshape(current_signals.shape[0], -1)
        else:
            input_signals = current_signals
            
        # Generate new signals
        new_signals = autoencoder.predict(input_signals)
        
        # Reshape back if needed
        if len(current_signals.shape) == 3:
            new_signals = new_signals.reshape(current_signals.shape)
            
        augmented_signals.append(new_signals)
        current_signals = new_signals
        
    return augmented_signals


def augment_dataset(autoencoder: Model, 
                   X: np.ndarray, 
                   y: np.ndarray, 
                   balance_classes: bool = False, 
                   num_iterations: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment a dataset using the autoencoder.
    
    Args:
        autoencoder: Trained autoencoder model
        X: Input features
        y: Target labels
        balance_classes: Whether to balance class distribution
        num_iterations: Number of iterations for augmentation
        
    Returns:
        Tuple of augmented features and labels
    """
    # Prepare data for augmentation
    X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
    
    if balance_classes:
        logger.info("Balancing classes through augmentation")
        # Find minority class
        class_counts = np.bincount(y)
        minority_class = np.argmin(class_counts)
        majority_class = np.argmax(class_counts)
        
        logger.info(f"Class distribution before balancing: {class_counts}")
        logger.info(f"Minority class: {minority_class}, count: {class_counts[minority_class]}")
        logger.info(f"Majority class: {majority_class}, count: {class_counts[majority_class]}")
        
        # Select samples from minority class
        minority_indices = np.where(y == minority_class)[0]
        X_minority = X_flat[minority_indices]
        
        # Calculate how many augmented samples needed
        num_to_generate = class_counts[majority_class] - class_counts[minority_class]
        logger.info(f"Generating {num_to_generate} new samples for minority class")
        
        # Generate augmented samples
        augmented_X = []
        augmented_y = []
        
        # Use iterative augmentation
        current_X = X_minority
        while len(augmented_X) < num_to_generate:
            new_X = autoencoder.predict(current_X)
            augmented_X.extend(new_X)
            augmented_y.extend([minority_class] * len(new_X))
            current_X = new_X
            
        # Trim to exact number needed
        augmented_X = augmented_X[:num_to_generate]
        augmented_y = augmented_y[:num_to_generate]
        
        # Combine original and augmented data
        X_augmented = np.vstack([X_flat, augmented_X])
        y_augmented = np.concatenate([y, augmented_y])
        
    else:
        logger.info(f"Augmenting entire dataset with {num_iterations} iterations")
        # Augment all samples
        augmented_signals = generate_augmented_signals(autoencoder, X_flat, num_iterations)
        
        # Combine original and all augmented versions
        X_augmented = np.vstack([X_flat] + augmented_signals)
        y_augmented = np.tile(y, num_iterations + 1)
    
    logger.info(f"Dataset size after augmentation: {X_augmented.shape[0]} samples")
    return X_augmented, y_augmented


def evaluate_classification(X_train: np.ndarray, 
                           y_train: np.ndarray, 
                           X_test: np.ndarray, 
                           y_test: np.ndarray) -> Dict[str, float]:
    """
    Train a classifier and evaluate its performance on seizure detection.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Training classifier on {X_train.shape[0]} samples")
    
    # Create a simple CNN classifier
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((178, 1), input_shape=(178,)),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred)
    }
    
    # Log results
    logger.info("\nClassification Results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall (Seizure Activity): {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    
    return metrics


def main():
    """Main function to run the autoencoder pipeline."""
    # Setup
    device = setup_device()
    
    # Load data
    X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test, scaler = load_train_hard_test_split()
    
    # Train autoencoder
    logger.info("Training autoencoder without dropout...")
    autoencoder_no_dropout = train_autoencoder(
        device, 
        X_train_scaled.reshape(-1, 178), 
        X_test_scaled.reshape(-1, 178),
        epochs=10
    )
    
    logger.info("\nTraining autoencoder with dropout...")
    autoencoder_with_dropout = train_autoencoder(
        device, 
        X_train_scaled.reshape(-1, 178), 
        X_test_scaled.reshape(-1, 178),
        epochs=10,
        one_d_dropout_rate=0.2,
        two_d_dropout_rate=0.3
    )
    
    # Visualize original vs reconstructed signals
    sample_idx = 0
    original_signal = X_test_scaled[sample_idx, :, 0]
    
    # Reconstruct with no-dropout model
    reconstructed_no_dropout = autoencoder_no_dropout.predict(
        X_test_scaled[sample_idx:sample_idx+1].reshape(1, 178)
    ).reshape(-1)
    
    # Reconstruct with dropout model
    reconstructed_with_dropout = autoencoder_with_dropout.predict(
        X_test_scaled[sample_idx:sample_idx+1].reshape(1, 178)
    ).reshape(-1)
    
    # Visualize
    visualize_signals(original_signal, reconstructed_no_dropout, 
                     "Original vs Reconstructed (No Dropout)")
    visualize_signals(original_signal, reconstructed_with_dropout, 
                     "Original vs Reconstructed (With Dropout)")
    
    # Data augmentation
    logger.info("\nAugmenting data to balance classes (no dropout)...")
    X_balanced_no_dropout, y_balanced_no_dropout = augment_dataset(
        autoencoder_no_dropout,
        X_train_scaled.reshape(-1, 178),
        y_train,
        balance_classes=True
    )
    
    logger.info("\nAugmenting data to balance classes (with dropout)...")
    X_balanced_with_dropout, y_balanced_with_dropout = augment_dataset(
        autoencoder_with_dropout,
        X_train_scaled.reshape(-1, 178),
        y_train,
        balance_classes=True
    )
    
    logger.info("\nAugmenting entire dataset (no dropout)...")
    X_augmented_no_dropout, y_augmented_no_dropout = augment_dataset(
        autoencoder_no_dropout,
        X_train_scaled.reshape(-1, 178),
        y_train,
        balance_classes=False,
        num_iterations=5
    )
    
    logger.info("\nAugmenting entire dataset (with dropout)...")
    X_augmented_with_dropout, y_augmented_with_dropout = augment_dataset(
        autoencoder_with_dropout,
        X_train_scaled.reshape(-1, 178),
        y_train,
        balance_classes=False,
        num_iterations=5
    )
    
    # Evaluate classification performance
    logger.info("\n--- Classification Performance ---")
    
    logger.info("\nBaseline (No Augmentation):")
    baseline_metrics = evaluate_classification(
        X_train_scaled.reshape(-1, 178),
        y_train,
        X_test_scaled.reshape(-1, 178),
        y_test
    )
    
    logger.info("\nBalanced Classes (No Dropout):")
    balanced_no_dropout_metrics = evaluate_classification(
        X_balanced_no_dropout,
        y_balanced_no_dropout,
        X_test_scaled.reshape(-1, 178),
        y_test
    )
    
    logger.info("\nBalanced Classes (With Dropout):")
    balanced_with_dropout_metrics = evaluate_classification(
        X_balanced_with_dropout,
        y_balanced_with_dropout,
        X_test_scaled.reshape(-1, 178),
        y_test
    )
    
    logger.info("\nFull Augmentation (No Dropout):")
    augmented_no_dropout_metrics = evaluate_classification(
        X_augmented_no_dropout,
        y_augmented_no_dropout,
        X_test_scaled.reshape(-1, 178),
        y_test
    )
    
    logger.info("\nFull Augmentation (With Dropout):")
    augmented_with_dropout_metrics = evaluate_classification(
        X_augmented_with_dropout,
        y_augmented_with_dropout,
        X_test_scaled.reshape(-1, 178),
        y_test
    )
    
    # Print summary table
    logger.info("\n--- Summary of Results ---")
    metrics = ['recall', 'f1', 'roc_auc', 'cohen_kappa']
    methods = [
        ('Baseline', baseline_metrics),
        ('Balanced (No Dropout)', balanced_no_dropout_metrics),
        ('Balanced (With Dropout)', balanced_with_dropout_metrics),
        ('Full Aug (No Dropout)', augmented_no_dropout_metrics),
        ('Full Aug (With Dropout)', augmented_with_dropout_metrics)
    ]
    
    logger.info(f"{'Method':<25} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10} {'Kappa':<10}")
    logger.info("-" * 65)
    
    for name, metric_dict in methods:
        values = [f"{metric_dict[m]:.4f}" for m in metrics]
        logger.info(f"{name:<25} {values[0]:<10} {values[1]:<10} {values[2]:<10} {values[3]:<10}")


if __name__ == "__main__":
    main()