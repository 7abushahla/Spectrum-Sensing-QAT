
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, LeakyReLU, ReLU, Flatten, Input, Dropout, Lambda, Reshape, MaxPooling2D, Add, Layer
from tensorflow.keras.models import Model
import argparse
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import pandas as pd
import tensorflow_model_optimization as tfmot

import logging
import tempfile
import os
import random
from datetime import datetime
import json
import shutil  # For copying files


from sklearn.model_selection import RepeatedKFold
from tensorflow.keras.models import load_model

# Suppress TensorFlow INFO, WARNING, and ERROR messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


DEVICE = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print(DEVICE)



# ================== Configuration Variables ==================
# Experiment Configuration
model_type = "ParallelCNN"      # Options: "DeepSense", "ParallelCNN"
N = 128                       # Options: 128, 32
training_type = "normal"      # Options: "normal", "QAT"
dataset = "LTE_SNR_10"                # Options: "SDR", "LTE"

# Reproducibility Settings
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Experiment Settings
N_FOLDS = 5
N_REPEATS = 1
EPOCHS = 150
BATCHSIZE = 256
# =============================================================



# ================== Naming Conventions ======================
# Define naming patterns based on configuration
metrics_filename = f"{model_type}_{N}_{training_type}_{dataset}_metrics.json"
best_overall_model_filename = f"{model_type}_{N}_{training_type}_{dataset}_best_overall_model.tflite"  # Updated to .tflite
tflite_model_filename_pattern = f"{model_type}_{N}_{training_type}_{dataset}_fold_{{fold_number}}_model.tflite"  # Dynamic naming per fold
# =============================================================



# ================== Directory Setup =========================
# Define base results directory
base_results_dir = "results"

# **Updated Directory Path to Include Dataset**
experiment_dir = os.path.join(
    base_results_dir,
    model_type,
    f"N{N}",
    training_type,
    dataset  # Added dataset to the path
)

# Subdirectories for models, logs, metrics, and plots
models_dir = os.path.join(experiment_dir, "models")
logs_dir = os.path.join(experiment_dir, "logs")
metrics_dir = os.path.join(experiment_dir, "metrics")
plots_dir = os.path.join(metrics_dir, "plots")  # Directory to save plots

# Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# **Optional: Print Directory Structure for Verification**
print(f"\nDirectories created:")
print(f" - Models: {models_dir}")
print(f" - Logs: {logs_dir}")
print(f" - Metrics: {metrics_dir}")
print(f" - Plots: {plots_dir}")

# List contents to confirm (should be empty initially)
print("\nContents of 'models_dir':", os.listdir(models_dir))
print("Contents of 'logs_dir':", os.listdir(logs_dir))
print("Contents of 'metrics_dir':", os.listdir(metrics_dir))
print("Contents of 'plots_dir':", os.listdir(plots_dir))
# =============================================================



# ================== Custom F1 Metric ==========================
@tf.keras.utils.register_keras_serializable()
def F1_Score(y_true, y_pred):
    # Cast inputs to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    # Compute true positives, false positives, and false negatives
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    # Compute micro-averaged precision and recall
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    # Compute F1-score
    f1 = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    return f1  # No need to take mean across classes (micro-averaged already)
# =============================================================

# ================== Custom Slice Layer ==========================
@tf.keras.utils.register_keras_serializable()
class SliceLayer(tf.keras.layers.Layer):
    def __init__(self, start, end, axis=-1, **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.start = start
        self.end = end
        self.axis = axis

    def call(self, inputs):
        slice_indices = [slice(None)] * len(inputs.shape)
        slice_indices[self.axis] = slice(self.start, self.end)
        return inputs[tuple(slice_indices)]

    def get_config(self):
        config = super(SliceLayer, self).get_config()
        config.update({
            'start': self.start,
            'end': self.end,
            'axis': self.axis
        })
        return config

# =============================================================

# ================== Data Loading =============================
dset = h5py.File("./lte_10_128_train.h5", 'r')
X = dset['X'][()]  # Original shape: (287971, 32, 2)
# Transpose the data; adjust axes as required.
X_transposed = np.transpose(X, (2, 1, 0))  # New shape: (471860, 128, 2)

# (Ensure the transpose order is correct for your application.)
y = dset['y'][()]  # Shape: (287971,)
y = y.T  # Transpose labels if needed

print(f"Training data shape: {X_transposed.shape}")
print(f"Sample Transposed Data (index 0):\n{X_transposed[0, :5, :]}")
print(f"Sample Transposed Data (index 1000):\n{X_transposed[1000, :5, :]}")
print(f"Sample Transposed Data (index 100000):\n{X_transposed[100000, :5, :]}")
print(f"Labels shape: {y.shape}")

# ================== Normalization and Statistics =============================
# Calculate mean and standard deviation across samples and time steps for each channel using float64
mean = np.mean(X_transposed, axis=(0, 1), dtype=np.float64)   # Mean for each channel
std = np.std(X_transposed, axis=(0, 1), dtype=np.float64)       # Standard deviation for each channel

print("\nComputed channel means (float64):", mean)
print("Computed channel standard deviations (float64):", std)

# Normalize the training data: (X - mean) / std
X_normalized = (X_transposed - mean) / std
X_normalized = X_normalized.astype(np.float64)

print("\nNormalized training data shape:", X_normalized.shape)
print("Sample normalized training data (first 10 time steps, all channels):")
print(X_normalized[0, :5, :])

# Compute and print overall normalization verification statistics
norm_mean = np.mean(X_normalized, axis=(0, 1), dtype=np.float64)
norm_std = np.std(X_normalized, axis=(0, 1), dtype=np.float64)

norm_mean_rounded = np.round(norm_mean, decimals=12)
print("\nNormalized channel means (rounded to 12 decimals):", np.abs(norm_mean_rounded))
print("Normalized channel standard deviations (float64):", norm_std)

# ----------------- Overall Data Range Calculation -----------------
# Assume channels are along the first dimension after transposition.
I_channel = X_normalized[:, :, 0].astype(np.float64)
Q_channel = X_normalized[:, :, 1].astype(np.float64)

# I channel statistics
I_min    = np.min(I_channel)
I_max    = np.max(I_channel)
I_mean   = np.mean(I_channel, dtype=np.float64)
I_median = np.median(I_channel)
I_std    = np.std(I_channel, dtype=np.float64)

# Percentile boundaries for I channel:
I_p0_1   = np.percentile(I_channel, 0.1)
I_p99_9  = np.percentile(I_channel, 99.9)
I_p0_01  = np.percentile(I_channel, 0.01)
I_p99_99 = np.percentile(I_channel, 99.99)

# Q channel statistics
Q_min = np.min(Q_channel)
Q_max = np.max(Q_channel)
Q_mean = np.mean(Q_channel, dtype=np.float64)
Q_median = np.median(Q_channel)
Q_std = np.std(Q_channel, dtype=np.float64)

Q_p0_1   = np.percentile(Q_channel, 0.1)
Q_p99_9  = np.percentile(Q_channel, 99.9)
Q_p0_01  = np.percentile(Q_channel, 0.01)
Q_p99_99 = np.percentile(Q_channel, 99.99)


print("\nI channel statistics:")
print(f"  Min: {I_min:.5f}, Max: {I_max:.5f}")
print(f"  Mean: {I_mean:.5f}, Median: {I_median:.5f}")
print(f"  Std: {I_std:.5f}")
print(f"  0.1th percentile: {I_p0_1:.5f}")
print(f"  99.9th percentile: {I_p99_9:.5f}")
print(f"  0.01th percentile: {I_p0_01:.5f}")
print(f"  99.99th percentile: {I_p99_99:.5f}")

print("\nQ channel statistics:")
print(f"  Min: {Q_min:.5f}, Max: {Q_max:.5f}")
print(f"  Mean: {Q_mean:.5f}, Median: {Q_median:.5f}")
print(f"  Std: {Q_std:.5f}")
print(f"  0.1th percentile: {Q_p0_1:.5f}")
print(f"  99.9th percentile: {Q_p99_9:.5f}")
print(f"  0.01th percentile: {Q_p0_01:.5f}")
print(f"  99.99th percentile: {Q_p99_99:.5f}")

overall_min = np.min(X_normalized)
overall_max = np.max(X_normalized)
print(f"\nOverall data range (min/max): {overall_min:.5f} to {overall_max:.5f}")

overall_p0_1 = np.percentile(X_normalized, 0.1)
overall_p99_9 = np.percentile(X_normalized, 99.9)
print(f"Overall data range (percentile-based, combined): {overall_p0_1:.5f} to {overall_p99_9:.5f}")

overall_new_min_0p1 = min(I_p0_1, Q_p0_1)
overall_new_max_99p9 = max(I_p99_9, Q_p99_9)
print(f"Overall new data range (percentile-based from channels, 0.1/99.9): {overall_new_min_0p1:.5f} to {overall_new_max_99p9:.5f}")

overall_new_min_0p01 = min(I_p0_01, Q_p0_01)
overall_new_max_99p99 = max(I_p99_99, Q_p99_99)
print(f"Overall new data range (percentile-based from channels, 0.01/99.99): {overall_new_min_0p01:.5f} to {overall_new_max_99p99:.5f}")



# ================== Load Testing Data (LTE) ========================
test_dset = h5py.File("./lte_10_128_test.h5", 'r')
X_test = test_dset['X'][()]   # Original shape: (samples, 32, 2) or similar
# Transpose test data to match training data orientation, e.g. (2, 128, samples)
X_test = np.transpose(X_test, (2, 1, 0))
y_test = test_dset['y'][()]   # Original shape: (samples,)
y_test = y_test.T            # Transpose labels if needed

print(f"Testing data shape: {X_test.shape}")
print(f"Sample Transposed Test Data (index 0):\n{X_test[0, :5, :]}")
print(f"Sample Transposed Test Data (index 1000):\n{X_test[1000, :5, :]}")
print(f"Sample Transposed Test Data (index 10000):\n{X_test[10000, :5, :]}")
print(f"Test Labels shape: {y_test.shape}")

# Ensure test data is in float64
X_test = X_test.astype(np.float64)

# Normalize the test data using the training mean and std
X_test_normalized = (X_test - mean) / std

print(f"Normalized testing data shape: {X_test_normalized.shape}")
print(f"Sample normalized testing data (first 10 samples):\n{X_test_normalized[0, :10, :]}")
# =============================================================

# ================== Cross-Validation Setup ===================
# Define cross-validation parameters
n_splits = N_FOLDS
n_repeats = N_REPEATS
random_state = SEED

# Initialize RepeatedKFold
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

# Initialize lists to store aggregated metrics
validation_precisions = []
validation_recalls = []
validation_f1_scores = []

test_precisions = []
test_recalls = []
test_f1_scores = []

# List to store per-channel metrics for each fold
channel_metrics_folds = []


# Variables to track the best overall model based on validation F1-score
best_overall_f1 = -1
best_overall_test_f1 = -1  # Initialize to track test F1-score for the best model
best_overall_model_path = os.path.join(models_dir, best_overall_model_filename)

# Variable to store the training history of the best overall model
best_overall_history = None

# Define TFLite model path pattern per fold
tflite_model_filename_pattern = f"{model_type}_{N}_{training_type}_{dataset}_fold_{{fold_number}}_model.tflite"

# Fold counter
total_folds = n_splits * n_repeats
fold_number = 1

# Get number of channels from test labels (assumes multi-label classification)
num_channels = y_test.shape[1]

# ================== Training Loop =============================
for train_index, val_index in rkf.split(X_normalized):
    print(f"\nStarting Fold {fold_number}/{total_folds}")
    
    # Split the data into training and validation sets for this fold
    X_train_fold, X_val_fold = X_normalized[train_index], X_normalized[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]
    
    print(f"Train shape: {X_train_fold.shape}, Validation shape: {X_val_fold.shape}")
    print(f"Train Labels shape: {y_train_fold.shape}, Validation Labels shape: {y_val_fold.shape}")
    
    # ================== Model Building ========================
    # Define model parameters
    M = y.shape[1]       # number of classes for multi-label classification
    N = X_normalized.shape[1]  # Number of I/Q samples
    n_channels = X_normalized.shape[2]  # Number of channels (I and Q)
    
    # Build the model
    inputs = Input(shape=(N, n_channels), name='input_layer')
    
    # Reshape input to fit Conv2D requirements: (1, dim, n_channels)
    reshaped_inputs = Reshape((1, N, n_channels), name='reshape')(inputs)
    
    # Split the input into two halves along the width (dim) dimension
    I1 = SliceLayer(start=0, end=int(N/2), axis=2, name='I1')(reshaped_inputs)  # First half
    I2 = SliceLayer(start=int(N/2), end=N, axis=2, name='I2')(reshaped_inputs)  # Second half

    # CNN1 Branch (Processing I1)
    # Here the kernel_size is (3,1) so that the convolution acts over the N dimension only.
    C1 = Conv2D(filters=8, kernel_size=(1, 3), strides=(1, 1), padding='valid')(I1)
    C1 = LeakyReLU(alpha=0.2)(C1)
    S1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(C1)

    C2 = Conv2D(filters=16, kernel_size=(1, 5), strides=(1, 2), padding='valid')(S1)
    C2 = LeakyReLU(alpha=0.2)(C2)
    S2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(C2)

    # CNN2 Branch (Processing I2)
    C3 = Conv2D(filters=8, kernel_size=(1, 3), strides=(1, 1), padding='valid')(I2)
    C3 = LeakyReLU(alpha=0.2)(C3)
    S3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(C3)

    C4 = Conv2D(filters=16, kernel_size=(1, 5), strides=(1, 2), padding='valid')(S3)
    C4 = LeakyReLU(alpha=0.2)(C4)
    S4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(C4)

    # Combining Outputs (element-wise summation)
    A1 = Add()([S2, S4])
    A1 = Flatten()(A1)

    # Fully Connected Layers
    F1 = Dense(64)(A1)
    F1 = LeakyReLU(alpha=0.2)(F1)
    F2 = Dense(M, activation='sigmoid')(F1)  # Multi-label classification

    # Creating Model
    model = Model(inputs=inputs, outputs=F2, name="ParallelCNN_2D")
    # =============================================================
    
    # ================== Model Compilation =====================
    # Compile the model
    adam = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        loss='binary_crossentropy', 
        optimizer=adam, 
        metrics=[
            tf.keras.metrics.Precision(name='Precision'), 
            tf.keras.metrics.Recall(name='Recall'), 
            F1_Score
        ]
    )
    # =============================================================
    
    # ================== Callbacks Setup =======================
    # Define a unique directory for TensorBoard logs per fold
    fold_log_dir = os.path.join(
        logs_dir, 
        f"fold_{fold_number}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    tensorboard_callback = TensorBoard(log_dir=fold_log_dir, histogram_freq=1)
    
    # Define ModelCheckpoint to save the best model based on validation F1-score
    checkpoint_filename = f"{model_type}_{N}_{training_type}_{dataset}_fold_{fold_number}_best_model.h5"
    checkpoint_path = os.path.join(models_dir, checkpoint_filename)
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_F1_Score',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    # Other callbacks
    # lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    # =============================================================
    
    # ================== Model Training =========================
    # Train the model
    history = model.fit(
        x=X_train_fold, 
        y=y_train_fold, 
        validation_data=(X_val_fold, y_val_fold),
        batch_size=BATCHSIZE, 
        epochs=EPOCHS, 
        verbose=1, 
        shuffle=True,  
        callbacks=[early_stopping, model_checkpoint, tensorboard_callback]
    )
    # =============================================================
    
    # ================== Model Evaluation ========================
    # Load the best model for this fold
    best_model_fold = load_model(
        checkpoint_path
    )
    
    # Convert the best model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model_fold)
    try:
        tflite_model = converter.convert()
        # Define the TFLite model path for this fold
        current_fold_tflite_path = os.path.join(
            models_dir, 
            tflite_model_filename_pattern.format(fold_number=fold_number)
        )
        # Save the TFLite model
        with open(current_fold_tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"Converted TFLite model saved to {current_fold_tflite_path}")
    except Exception as e:
        print(f"Fold {fold_number} - TFLite conversion failed with error: {e}")
        current_fold_tflite_path = None
    
    if current_fold_tflite_path and os.path.exists(current_fold_tflite_path):
        # Load the TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=current_fold_tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Function to run inference with TFLite
        def run_inference_tflite(interpreter, X):
            # Assuming single input
            input_index = input_details[0]['index']
            output_index = output_details[0]['index']
            
            # Ensure X has the right shape (batch_size, dim, n_channels)
            if len(X.shape) == 2:
                X = np.expand_dims(X, axis=0)
            
            interpreter.set_tensor(input_index, X.astype(np.float32))
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)
            return output
        
        # Evaluate the model on the test set
        y_pred = []
        batch_size = 1  # Adjust as needed
        
        for i in range(0, len(X_test_normalized), batch_size):
            X_batch = X_test_normalized[i:i + batch_size]
            predictions = run_inference_tflite(interpreter, X_batch)
            y_pred.append(predictions)
        
        # Concatenate all predictions
        y_pred = np.concatenate(y_pred, axis=0)
        
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Compute Precision, Recall, and F1-Score using scikit-learn
        precision = precision_score(y_test, y_pred_binary, average='micro', zero_division=0)
        recall = recall_score(y_test, y_pred_binary, average='micro', zero_division=0)
        f1 = f1_score(y_test, y_pred_binary, average='micro', zero_division=0)
        
        print(f"Fold {fold_number} - Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        
        # Append test metrics to the aggregated lists
        test_precisions.append(precision)
        test_recalls.append(recall)
        test_f1_scores.append(f1)
        
        # ----------------- Compute Per-Channel Metrics ------------------
        fold_channel_metrics = []  # This fold's metrics for each channel
        for channel in range(num_channels):
            precision_ch = precision_score(
                y_test[:, channel], y_pred_binary[:, channel],
                average='binary', zero_division=0)
            recall_ch = recall_score(
                y_test[:, channel], y_pred_binary[:, channel],
                average='binary', zero_division=0)
            f1_ch = f1_score(
                y_test[:, channel], y_pred_binary[:, channel],
                average='binary', zero_division=0)
            fold_channel_metrics.append([precision_ch, recall_ch, f1_ch])
        # Convert to NumPy array (shape: [num_channels, 3]) and store it
        channel_metrics_folds.append(np.array(fold_channel_metrics))
    else:
        print(f"Fold {fold_number} - Skipping test evaluation due to TFLite conversion failure.")
    
    # ================== Best Overall Model ======================
    # Evaluate validation F1-score from training history
    if 'history' in locals():
        if 'val_F1_Score' in history.history:
            val_f1 = max(history.history['val_F1_Score'])
            # Append validation metrics
            val_precision = max(history.history['Precision'])
            val_recall = max(history.history['Recall'])
            validation_precisions.append(val_precision)
            validation_recalls.append(val_recall)
            validation_f1_scores.append(val_f1)
        else:
            print(f"Fold {fold_number} - 'val_F1_Score' not found in history.")
            val_f1 = -1  # Assign a default value or handle appropriately
    else:
        print(f"Fold {fold_number} - 'history' object not found.")
        val_f1 = -1  # Assign a default value or handle appropriately
    
    # Check if this fold has the best validation F1-score
    if val_f1 > best_overall_f1 and current_fold_tflite_path:
        best_overall_f1 = val_f1
        best_overall_test_f1 = f1  # Update test F1-score for the best model
        best_overall_history = history  # Store the training history of the best model
        
        # Copy the current fold's TFLite model to the best_overall_model_path
        shutil.copy(current_fold_tflite_path, best_overall_model_path)
        print(f"New best overall model found in Fold {fold_number} with Validation F1-Score: {val_f1:.4f}")
    elif val_f1 > best_overall_f1 and not current_fold_tflite_path:
        print(f"Fold {fold_number} has a better validation F1-Score ({val_f1:.4f}) but TFLite conversion failed. Not updating the best model.")
    # =============================================================
    
    fold_number += 1
# =============================================================



# ================== Aggregating Metrics ======================
# Compute average and standard deviation for validation metrics
avg_val_precision = np.mean(validation_precisions) if validation_precisions else 0
std_val_precision = np.std(validation_precisions) if validation_precisions else 0

avg_val_recall = np.mean(validation_recalls) if validation_recalls else 0
std_val_recall = np.std(validation_recalls) if validation_recalls else 0

avg_val_f1 = np.mean(validation_f1_scores) if validation_f1_scores else 0
std_val_f1 = np.std(validation_f1_scores) if validation_f1_scores else 0

# Compute average and standard deviation for test metrics
avg_test_precision = np.mean(test_precisions) if test_precisions else 0
std_test_precision = np.std(test_precisions) if test_precisions else 0

avg_test_recall = np.mean(test_recalls) if test_recalls else 0
std_test_recall = np.std(test_recalls) if test_recalls else 0

avg_test_f1 = np.mean(test_f1_scores) if test_f1_scores else 0
std_test_f1 = np.std(test_f1_scores) if test_f1_scores else 0
# =============================================================

# ================== Aggregate Per-Channel Metrics ============================
# Stack the per-fold metrics into an array of shape (num_folds, num_channels, 3)
if channel_metrics_folds:
    all_channel_metrics = np.stack(channel_metrics_folds, axis=0)
    # Compute mean and std across folds for each channel (axis=0)
    mean_channel_metrics = np.mean(all_channel_metrics, axis=0)  # shape: [num_channels, 3]
    std_channel_metrics = np.std(all_channel_metrics, axis=0)
else:
    mean_channel_metrics = np.zeros((num_channels, 3))
    std_channel_metrics = np.zeros((num_channels, 3))

# Also compute micro-average metrics across folds
mean_micro_precision = np.mean(test_precisions) if test_precisions else 0
mean_micro_recall = np.mean(test_recalls) if test_recalls else 0
mean_micro_f1 = np.mean(test_f1_scores) if test_f1_scores else 0
# =============================================================

# ================== Metrics Summary ==========================
# Prepare the metrics summary
metrics_summary = {
    'cross_validation': {
        'total_folds': total_folds,
        'validation_metrics': {
            'precision': {
                'mean': float(avg_val_precision),
                'std': float(std_val_precision)
            },
            'recall': {
                'mean': float(avg_val_recall),
                'std': float(std_val_recall)
            },
            'f1_score': {
                'mean': float(avg_val_f1),
                'std': float(std_val_f1)
            }
        },
        'test_metrics': {
            'precision': {
                'mean': float(avg_test_precision),
                'std': float(std_test_precision)
            },
            'recall': {
                'mean': float(avg_test_recall),
                'std': float(std_test_recall)
            },
            'f1_score': {
                'mean': float(avg_test_f1),
                'std': float(std_test_f1)
            }
        },
        # Add per-channel metrics after test metrics
        'per_channel_metrics': {
            **{
                f'Channel-{i+1}': {
                    'Precision': {'mean': float(mean_channel_metrics[i, 0]),
                                  'std': float(std_channel_metrics[i, 0])},
                    'Recall': {'mean': float(mean_channel_metrics[i, 1]),
                               'std': float(std_channel_metrics[i, 1])},
                    'F1-score': {'mean': float(mean_channel_metrics[i, 2]),
                                 'std': float(std_channel_metrics[i, 2])},
                    'Occurrences': int(y_test.sum(axis=0)[i])
                }
                for i in range(num_channels)
            },
            'Micro Average': {
                'Precision': float(mean_micro_precision),
                'Recall': float(mean_micro_recall),
                'F1-score': float(mean_micro_f1),
                'Occurrences': int(np.sum(y_test.sum(axis=0)))
            }
        },
        # Place the best overall model at the end
        'best_overall_model': {
            'model_path': best_overall_model_path,
            'validation_f1_score': float(best_overall_f1),
            'test_f1_score': float(best_overall_test_f1) if best_overall_test_f1 != -1 else None
        }
    }
}
# =============================================================

# ================== Save Metrics to JSON =====================
# Define the metrics file path
metrics_file_path = os.path.join(metrics_dir, metrics_filename)

# Save the metrics to a JSON file
with open(metrics_file_path, 'w') as json_file:
    json.dump(metrics_summary, json_file, indent=4)

print(f"\nFinal aggregated metrics have been saved to '{metrics_file_path}'.")
print(f"Best overall TFLite model saved at: '{best_overall_model_path}'")
# =============================================================


# ================== Generate and Save Plots for Best Overall Model ============
if best_overall_history is not None:
    print("\nGenerating plots for the best overall model...")
    
    # Define plot filenames with fixed names
    loss_plot_filename = "Best_Model_Loss.png"
    precision_plot_filename = "Best_Model_Precision.png"
    recall_plot_filename = "Best_Model_Recall.png"
    f1_score_plot_filename = "Best_Model_F1_Score.png"
    
    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(best_overall_history.history['loss'], label='Training Loss')
    plt.plot(best_overall_history.history['val_loss'], label='Validation Loss')
    plt.title('Best Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, loss_plot_filename))
    plt.close()
    
    # Plot Precision
    plt.figure(figsize=(8, 6))
    plt.plot(best_overall_history.history['Precision'], label='Training Precision')
    plt.plot(best_overall_history.history['val_Precision'], label='Validation Precision')
    plt.title('Best Model Precision Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, precision_plot_filename))
    plt.close()
    
    # Plot Recall
    plt.figure(figsize=(8, 6))
    plt.plot(best_overall_history.history['Recall'], label='Training Recall')
    plt.plot(best_overall_history.history['val_Recall'], label='Validation Recall')
    plt.title('Best Model Recall Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, recall_plot_filename))
    plt.close()
    
    # Plot F1-Score
    plt.figure(figsize=(8, 6))
    plt.plot(best_overall_history.history['F1_Score'], label='Training F1 Score')
    plt.plot(best_overall_history.history['val_F1_Score'], label='Validation F1 Score')
    plt.title('Best Model F1 Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f1_score_plot_filename))
    plt.close()
    
    print(f"Standard training plots for the best overall model have been saved to '{plots_dir}'.")
    
    # ================== Generate Heatmap for Per-Channel Metrics ====================
    print("\nGenerating per-channel heatmap for the best overall model...")
    
    # Load the best overall TFLite model
    interpreter = tf.lite.Interpreter(model_path=best_overall_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Run inference on the entire test set using the best overall model
    y_pred_best = []
    for i in range(0, len(X_test_normalized), 1):
        X_batch = X_test_normalized[i:i + 1]
        interpreter.set_tensor(input_details[0]['index'], X_batch.astype(np.float32))
        interpreter.invoke()
        y_pred_best.append(interpreter.get_tensor(output_details[0]['index']))
    y_pred_best = np.concatenate(y_pred_best, axis=0)
    y_pred_best_binary = (y_pred_best > 0.5).astype(int)
    
    # Compute per-channel metrics for the best overall model
    best_model_channel_metrics = []
    for channel in range(num_channels):
        precision_ch = precision_score(
            y_test[:, channel], y_pred_best_binary[:, channel],
            average='binary', zero_division=0)
        recall_ch = recall_score(
            y_test[:, channel], y_pred_best_binary[:, channel],
            average='binary', zero_division=0)
        f1_ch = f1_score(
            y_test[:, channel], y_pred_best_binary[:, channel],
            average='binary', zero_division=0)
        best_model_channel_metrics.append([precision_ch, recall_ch, f1_ch])
    best_model_channel_metrics = np.array(best_model_channel_metrics)
    
    # Create a heatmap (reversed order so Channel-4 appears on top, for example)
    fig, ax = plt.subplots(figsize=(7, 8))
    cmap = sns.color_palette("Blues", as_cmap=True)  # "crest" provides a smooth blue-green gradient

    sns.heatmap(best_model_channel_metrics[::-1], annot=True, fmt=".4f", cmap=cmap,
                xticklabels=['Precision', 'Recall', 'F1-score'],
                yticklabels=[f'Channel-{i+1} ({int(y_test.sum(axis=0)[i])})' for i in range(num_channels-1, -1, -1)],
                cbar=True)
    plt.title("SDR Dataset Performance Metrics (Best Overall Model)")
    plt.ylabel("Channels (# of Occurrences)")
    plt.xlabel("Metrics")
    
    # Adjust layout so that nothing is cut off
    plt.tight_layout()
    heatmap_path = os.path.join(plots_dir, "Best_Model_Per_Channel_Heatmap.png")
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to '{heatmap_path}'.")
else:
    print("\nNo best overall model was identified. Plot generation skipped.")
# =============================================================





