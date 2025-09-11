import pandas as pd
import pickle
import shap
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score, 
                             confusion_matrix, roc_auc_score, roc_curve,
                             mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import os
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

class NeuralNetworkModel:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)
        
    def build_model(self, input_shape):
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.4),  # Increased dropout for better regularization
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        return model

class TrainingTracker:
    """Class to track training and validation metrics during model training."""
    def __init__(self):
        self.history = defaultdict(list)
        
    def update(self, history):
        for key, values in history.history.items():
            self.history[key].extend(values)
    
    def plot_learning_curves(self, output_dir):
        """Plot training vs validation accuracy and loss curves."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot accuracy curves
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training vs Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/accuracy_curve.png")
        plt.close()
        
        # Plot loss curves
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/loss_curve.png")
        plt.close()
        
        # Plot AUC curves
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['auc'], label='Training AUC')
        plt.plot(self.history['val_auc'], label='Validation AUC')
        plt.title('Training vs Validation AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/auc_curve.png")
        plt.close()

def train_model(train_csv, val_csv, model_output):
    """Train and evaluate the neural network model."""
    print("Loading training data...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # Prepare features and target
    non_feature_cols = ['category', 'rating', 'label', 'text_', 'cleaned_text']
    feature_cols = [col for col in train_df.columns if col not in non_feature_cols]
    
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    
    # Map labels
    label_map = {'CG': 1, 'OR': 0}  # CG=Fake, OR=Genuine
    y_train = train_df['label'].map(label_map).values
    y_val = val_df['label'].map(label_map).values
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Initialize model
    input_shape = X_train_scaled.shape[1]
    model_wrapper = NeuralNetworkModel(input_shape)
    model = model_wrapper.model
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_auc', mode='max'),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Initialize tracker
    tracker = TrainingTracker()
    
    # Train model
    print("\nTraining neural network...")
    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Update tracker
    tracker.update(history)
    
    # Final evaluation
    y_pred = (model.predict(X_val_scaled) > 0.5).astype(int)
    y_proba = model.predict(X_val_scaled)
    
    # Calculate metrics
    train_metrics = model.evaluate(X_train_scaled, y_train, verbose=0)
    val_metrics = model.evaluate(X_val_scaled, y_val, verbose=0)
    
    # Print metrics
    print("\n=== Validation Set Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_val, y_pred)*100:.2f}%")
    print(f"AUC-ROC: {roc_auc_score(y_val, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Save metrics to file
    os.makedirs("results", exist_ok=True)
    with open("results/validation_metrics.txt", "w") as f:
        f.write("Validation Set Metrics:\n")
        f.write(f"Accuracy: {accuracy_score(y_val, y_pred)*100:.2f}%\n")
        f.write(f"AUC-ROC: {roc_auc_score(y_val, y_proba):.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_val, y_pred))
    
    # Plot learning curves
    tracker.plot_learning_curves("results/validation_plots")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_val, y_proba):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("results/validation_plots/roc_curve.png")
    plt.close()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1], ['Genuine', 'Fake'])
    plt.yticks([0, 1], ['Genuine', 'Fake'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='red')
    plt.savefig("results/validation_plots/confusion_matrix.png")
    plt.close()
    
    # SHAP explainability - Fixed implementation
    try:
        print("\nGenerating SHAP explanations...")
        sample_size = min(100, X_val_scaled.shape[0])
        X_sample = X_val_scaled[:sample_size]
        
        # Create a SHAP explainer
        explainer = shap.DeepExplainer(
            model,
            X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]
        )
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, we get a list of arrays (one per class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take values for positive class
        
        # Summary dot plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            features=X_sample,
            feature_names=feature_cols,
            plot_type="dot",
            show=False,
            max_display=min(20, len(feature_cols))
        )
        plt.tight_layout()
        plt.savefig("results/shap_summary.png", bbox_inches='tight', dpi=300)
        plt.close()

        # Summary bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            features=X_sample,
            feature_names=feature_cols,
            plot_type="bar",
            show=False,
            max_display=min(20, len(feature_cols))
        )
        plt.tight_layout()
        plt.savefig("results/shap_bar_plot.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        print("✅ SHAP plots generated successfully")
    
    except Exception as e:
        print(f"\nWarning: SHAP visualization failed - {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Save model with correct extension
    model_output = model_output.replace('.pkl', '.h5')  # Ensure correct extension
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    model.save(model_output)
    
    # Save scaler and metadata separately
    metadata_path = model_output.replace('.h5', '_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'feature_names': feature_cols,
            'label_map': label_map,
            'metrics': {
                'accuracy': accuracy_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_proba),
                'classification_report': classification_report(y_val, y_pred, output_dict=True),
                'training_history': tracker.history
            }
        }, f)
    print(f"\n✅ Model saved to {model_output}")
    print(f"✅ Metadata saved to {metadata_path}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("results/validation_plots", exist_ok=True)
    
    train_model(
        train_csv="data/analysis/train_analysis.csv",
        val_csv="data/analysis/val_analysis.csv",
        model_output="data/models/fake_review_model.h5"  # Correct extension
    )