#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, LeakyReLU, Flatten, Input, Dropout, Lambda, Reshape, MaxPooling2D, ReLU
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


# In[15]:


# Suppress TensorFlow INFO, WARNING, and ERROR messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[16]:


DEVICE = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print(DEVICE)


# In[17]:


# ================== Configuration Variables ==================
# Experiment Configuration
model_type = "DeepSense"      # Options: "DeepSense", "ParallelCNN"
N = 32                       # Options: 128, 32
training_type = "QAT"         # Fixed to "QAT" for this script

# Reproducibility Settings
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Experiment Settings
N_FOLDS = 5
N_REPEATS = 3
EPOCHS = 100
BATCHSIZE = 256
# =============================================================


# In[18]:


# ================== Naming Conventions ======================
# Define naming patterns based on configuration
metrics_filename = f"{model_type}_{N}_{training_type}_metrics.json"
best_overall_model_filename = f"{model_type}_{N}_{training_type}_best_overall_model_INT8.tflite"  # Updated to .tflite
tflite_model_filename_pattern = f"{model_type}_{N}_{training_type}_fold_{{fold_number}}_model_INT8.tflite"  # Dynamic naming per fold
# =============================================================


# In[19]:


# ================== Directory Setup =========================
# Define base results directory
base_results_dir = "results"

# Construct the directory path
experiment_dir = os.path.join(
    base_results_dir,
    model_type,
    f"N{N}",
    training_type
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

# Verification
print(f"Directories created:")
print(f" - Models: {models_dir}")
print(f" - Logs: {logs_dir}")
print(f" - Metrics: {metrics_dir}")
print(f" - Plots: {plots_dir}")

# List contents to confirm
print("\nContents of 'models_dir':", os.listdir(models_dir))
print("Contents of 'logs_dir':", os.listdir(logs_dir))
print("Contents of 'metrics_dir':", os.listdir(metrics_dir))
print("Contents of 'plots_dir':", os.listdir(plots_dir))
# =============================================================


# In[20]:


# ================== Custom F1 Metric ==========================
def F1_Score(y_true, y_pred):
    # Cast the y_true and y_pred to the right shape (binary for multi-label classification)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    
    # Precision calculation
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32), axis=0)
    predicted_positives = tf.reduce_sum(tf.cast(y_pred, tf.float32), axis=0)
    actual_positives = tf.reduce_sum(tf.cast(y_true, tf.float32), axis=0)

    precision = tp / (predicted_positives + tf.keras.backend.epsilon())
    recall = tp / (actual_positives + tf.keras.backend.epsilon())

    # F1 calculation
    f1 = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
    
    # Mean of F1 across all classes
    return tf.reduce_mean(f1)
# =============================================================


# In[21]:


# ================== Data Loading =============================
# Load training data from the .h5 file
dset = h5py.File("./sdr_wifi_train_32buf.hdf5", 'r')
X = dset['X'][()]  # Shape: (287971, 32, 2)
y = dset['y'][()]  # Shape: (287971,)

print(f"Training data shape: {X.shape}")
print(f"Sample training data:\n{X[0, :5, :2]}")  # Display first 5 samples

print(f"Labels shape: {y.shape}")
print(f"Sample labels (first sample):\n{y[0]}")  # Display first label

# Normalize the training data
# Calculate mean and standard deviation across the dataset for each channel
mean = np.mean(X, axis=(0, 1))  # Mean for each channel
std = np.std(X, axis=(0, 1))    # Standard deviation for each channel

# Perform normalization: (X - mean) / std
X_normalized = (X - mean) / std

# Print normalized data for verification
print(f"Training data shape: {X_normalized.shape}")
print(f"Sample normalized training data:\n{X_normalized[0, :5, :]}")
# =============================================================


# In[22]:


# ================== Load Testing Data ========================
# Load testing data from the .h5 file
test_dset = h5py.File("./sdr_wifi_test_32buf.hdf5", 'r')
X_test = test_dset['X'][()]
y_test = test_dset['y'][()]

print(f"Testing data shape: {X_test.shape}")
print(f"Sample testing data (first 10 samples):\n{X_test[0, :5, :]}")

print(f"Test Labels shape: {y_test.shape}")
print(f"Sample test labels (first sample):\n{y_test[0]}")

# Normalize the test data using training mean and std
X_test_normalized = (X_test - mean) / std

print(f"Normalized testing data shape: {X_test_normalized.shape}")
print(f"Sample normalized testing data (first 10 samples):\n{X_test_normalized[0, :5, :]}")
# =============================================================


# In[23]:


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

# Variables to track the best overall model based on validation F1-score
best_overall_f1 = -1
best_overall_test_f1 = -1  # Initialize to track test F1-score for the best model
best_overall_model_path = os.path.join(models_dir, best_overall_model_filename)

# Variable to store the training history of the best overall model
best_overall_history = None

# Define TFLite model path pattern per fold
tflite_model_filename_pattern = f"{model_type}_{N}_{training_type}_fold_{{fold_number}}_model_INT8.tflite"

# Fold counter
total_folds = n_splits * n_repeats
fold_number = 1
# =============================================================

# ================== Representative Dataset =====================
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(X_normalized).batch(1).take(100):
        yield [input_value]
# =============================================================

# ================== Training Loop =============================
for train_index, val_index in rkf.split(X_normalized):
    print(f"\nStarting Fold {fold_number}/{total_folds}")
    
    # Split the data into training and validation sets for this fold
    X_train_fold, X_val_fold = X_normalized[train_index], X_normalized[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]
    
    print(f"Train shape: {X_train_fold.shape}, Validation shape: {X_val_fold.shape}")
    
    # ================== Model Building ========================
    # Define model parameters
    n_classes = y.shape[1]       # number of classes for SDR case
    dim = X_normalized.shape[1]  # Number of I/Q samples being taken as input
    n_channels = X_normalized.shape[2]  # Number of channels (I and Q)
    
    # Build the model
    inputs = Input(shape=(dim, n_channels), dtype=tf.float32, name='input_layer')
    
    # Reshape input to fit Conv2D requirements: (1, dim, n_channels)
    reshaped_inputs = Reshape((1, dim, n_channels))(inputs)
    
    # First Conv stack
    x = Conv2D(16, (1, 3), name='conv1')(reshaped_inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(16, (1, 3), name='conv2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), name='pool1')(x)
    x = Dropout(0.3)(x)  # Dropout added after the first stack
    
    # Second Conv stack
    x = Conv2D(32, (1, 5), name='conv3')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(32, (1, 5), name='conv4')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), name='pool2')(x)
    x = Dropout(0.3)(x)  # Dropout added after the second stack
    
    # Fully connected layer
    x = Flatten()(x)
    x = Dense(64, name='dense1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Output layer
    outputs = Dense(n_classes, activation='sigmoid', name='out')(x)
    
    # Create base model
    model = Model(inputs=inputs, outputs=outputs)
    # =============================================================
    
    # ================== Quantization Aware Training ============
    # Apply QAT to the model
    qat_model = tfmot.quantization.keras.quantize_model(model)
    print("Converted model to QAT.")
    # =============================================================
    
    # ================== Model Compilation =====================
    # Compile the QAT model
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    qat_model.compile(
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
    
    # Define ModelCheckpoint to save the best QAT model based on validation F1-score
    checkpoint_filename = f"{model_type}_{N}_{training_type}_fold_{fold_number}_best_model.h5"
    checkpoint_path = os.path.join(models_dir, checkpoint_filename)
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_F1_Score',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    # Other callbacks
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    # =============================================================
    
    # ================== Model Training =========================
    # Train the QAT model
    history = qat_model.fit(
        x=X_train_fold, 
        y=y_train_fold, 
        validation_data=(X_val_fold, y_val_fold),
        batch_size=BATCHSIZE, 
        epochs=EPOCHS, 
        verbose=1, 
        shuffle=True,  
        callbacks=[lr_scheduler, early_stopping, model_checkpoint, tensorboard_callback]
    )
    # =============================================================
    
    # ================== Model Evaluation ========================
    # Load the best QAT model for this fold within quantize_scope
    try:
        with tfmot.quantization.keras.quantize_scope():
            best_model_fold = load_model(
                checkpoint_path, 
                custom_objects={'F1_Score': F1_Score}
            )
    except ValueError as e:
        print(f"Fold {fold_number} - Error loading model: {e}")
        best_model_fold = None
    
    if best_model_fold is not None:
        # ================== TFLite Conversion ======================
        # Initialize the TFLite converter with the quantization-aware model
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model_fold)
        
        # Set optimization flag to enable quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Provide the representative dataset for calibration
        converter.representative_dataset = representative_data_gen
        
        # Specify that we want int8 operations for inputs and outputs
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8  # Set to int8
        
        # Perform the conversion
        try:
            quantized_tflite_model = converter.convert()
            # Define the TFLite model path for this fold
            current_fold_tflite_path = os.path.join(
                models_dir, 
                tflite_model_filename_pattern.format(fold_number=fold_number)
            )
            # Save the quantized model to a file
            with open(current_fold_tflite_path, "wb") as f:
                f.write(quantized_tflite_model)
            print(f"Quantized TFLite model saved to {current_fold_tflite_path}")
        except ValueError as e:
            print(f"Fold {fold_number} - TFLite conversion failed with error: {e}")
            quantized_tflite_model = None  # Handle conversion failure
        
        # If conversion was successful, proceed to evaluation
        if quantized_tflite_model:
            # Load the TFLite model and allocate tensors
            interpreter = tf.lite.Interpreter(model_path=current_fold_tflite_path)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Function to quantize inputs from FLOAT32 to INT8
            def quantize_input(X, scale, zero_point):
                return np.round(X / scale + zero_point).astype(np.int8)
            
            # Retrieve scale and zero point from the model's input details (calculated during training quantization)
            input_scale, input_zero_point = input_details[0]['quantization']
            
            # Retrieve scale and zero point for outputs
            output_scale, output_zero_point = output_details[0]['quantization']
            
            # Define inference function
            def run_inference_tflite(X):
                # Quantize input
                X_int8 = quantize_input(X, input_scale, input_zero_point)
                interpreter.set_tensor(input_details[0]['index'], X_int8)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                return output
            
            # Evaluate the model on the test set
            y_pred = []
            batch_size = 1  # Adjust batch size as needed
            
            for i in range(0, len(X_test_normalized), batch_size):
                X_batch = X_test_normalized[i:i + batch_size]
                
                # Add batch dimension if necessary
                if len(X_batch.shape) == 2:
                    X_batch = np.expand_dims(X_batch, axis=0)
                
                # Run inference
                predictions = run_inference_tflite(X_batch)
                
                # Store predictions
                y_pred.append(predictions)
            
            # Concatenate predictions
            y_pred = np.concatenate(y_pred, axis=0)
            
            # Dequantize outputs
            y_pred_float32 = (y_pred.astype(np.float32) - output_zero_point) * output_scale
            
            # Convert probabilities to binary
            y_pred_binary = (y_pred_float32 > 0.5).astype(int)
            
            # Compute Precision, Recall, and F1-Score
            precision = precision_score(y_test, y_pred_binary, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred_binary, average='macro', zero_division=0)
            f1_value = f1_score(y_test, y_pred_binary, average='macro', zero_division=0)
            
            print(f"Fold {fold_number} - Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_value:.4f}")
            
            # Append test metrics to the aggregated lists
            test_precisions.append(precision)
            test_recalls.append(recall)
            test_f1_scores.append(f1_value)
    else:
        print(f"Fold {fold_number} - Skipping TFLite conversion and test evaluation due to model loading failure.")
    
    # =============================================================
    
    # ================== Best Overall Model ======================
    # Extract the best validation F1-score from the training history
    if 'history' in locals():
        if 'val_F1_Score' in history.history:
            val_f1 = max(history.history['val_F1_Score'])
            # Append validation metrics
            validation_precisions.append(max(history.history['val_Precision']))
            validation_recalls.append(max(history.history['val_Recall']))
            validation_f1_scores.append(val_f1)
        else:
            print(f"Fold {fold_number} - 'val_F1_Score' not found in history.")
            val_f1 = -1  # Assign a default value or handle appropriately
    else:
        print(f"Fold {fold_number} - 'history' object not found.")
        val_f1 = -1  # Assign a default value or handle appropriately
    
    # Check if this fold has the best validation F1-score
    if val_f1 > best_overall_f1 and quantized_tflite_model:
        best_overall_f1 = val_f1
        best_overall_test_f1 = f1_value  # Update test F1-score for the best model
        best_overall_history = history  # Store the training history of the best model
        
        # Copy the current fold's TFLite model to the best_overall_model_path
        shutil.copy(current_fold_tflite_path, best_overall_model_path)
        print(f"New best overall model found in Fold {fold_number} with Validation F1-Score: {val_f1:.4f}")
    elif val_f1 > best_overall_f1 and not quantized_tflite_model:
        print(f"Fold {fold_number} has a better validation F1-Score ({val_f1:.4f}) but TFLite conversion failed. Not updating the best model.")
    # =============================================================
    
    fold_number += 1
# =============================================================


# In[ ]:


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
        'best_overall_model': {
            'model_path': best_overall_model_path,
            'validation_f1_score': float(best_overall_f1),
            'test_f1_score': float(best_overall_test_f1) if best_overall_test_f1 != -1 else None
        }
    }
}

# print("\nCross-Validation Metrics Summary:")
# print(json.dumps(metrics_summary, indent=4))
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
    
    print(f"Plots for the best overall model have been saved to '{plots_dir}'.")
else:
    print("\nNo best overall model was identified. Plot generation skipped.")
# =============================================================


# In[ ]:




