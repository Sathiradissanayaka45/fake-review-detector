import pandas as pd
import numpy as np
import pickle
import os
import argparse
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (classification_report, accuracy_score, 
                            confusion_matrix, roc_auc_score, mean_squared_error, r2_score)

def load_model_and_metadata(model_path, metadata_path):
    """Load trained model and associated metadata."""
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return model, metadata

def predict(model, metadata, input_csv, output_csv=None):
    """Generate predictions for the input data."""
    print(f"Loading input data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Extract features
    non_feature_cols = ['category', 'rating', 'label', 'text_', 'cleaned_text']
    all_features = metadata['feature_names']
    
    # Check if all required features are present
    missing_features = [feat for feat in all_features if feat not in df.columns]
    if missing_features:
        raise ValueError(f"Input data is missing required features: {missing_features}")
    
    # Get feature data
    X = df[all_features]
    
    # Apply scaling
    X_scaled = metadata['scaler'].transform(X)
    
    # Apply feature selection if available
    if 'selected_indices' in metadata:
        X_selected = X_scaled[:, metadata['selected_indices']]
    else:
        X_selected = X_scaled
    
    # Generate predictions
    print("Generating predictions...")
    y_proba = model.predict(X_selected)
    
    # Ensure probabilities are properly shaped
    if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
        y_proba = y_proba[:, 1]
    else:
        y_proba = y_proba.flatten()
    
    y_pred = (y_proba > 0.5).astype(int)
    
    # Map predictions back to original labels if desired
    label_map = metadata['label_map']
    rev_label_map = {v: k for k, v in label_map.items()}
    y_pred_labels = [rev_label_map[pred] for pred in y_pred]
    
    # Add predictions to dataframe
    df['predicted_probability'] = y_proba
    df['predicted_label_numeric'] = y_pred
    df['predicted_label'] = y_pred_labels
    
    # Evaluate if true labels are available
    if 'label' in df.columns:
        y_true = df['label'].map(label_map)
        accuracy = accuracy_score(y_true, y_pred)
        
        try:
            auc = roc_auc_score(y_true, y_proba)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            print("\n=== Prediction Metrics ===")
            print(f"Accuracy: {accuracy*100:.2f}%")
            print(f"AUC-ROC: {auc:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R² Score: {r2:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred))
            
            # Save metrics to file
            os.makedirs("results/prediction_metrics", exist_ok=True)
            with open("results/prediction_metrics/metrics.txt", "w") as f:
                f.write("Prediction Metrics:\n")
                f.write(f"Accuracy: {accuracy*100:.2f}%\n")
                f.write(f"AUC-ROC: {auc:.4f}\n")
                f.write(f"MSE: {mse:.4f}\n")
                f.write(f"RMSE: {rmse:.4f}\n")
                f.write(f"R² Score: {r2:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(classification_report(y_true, y_pred))
            
            # Plot confusion matrix
            os.makedirs("results/prediction_plots", exist_ok=True)
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks([0, 1], ['Genuine', 'Fake'])
            plt.yticks([0, 1], ['Genuine', 'Fake'])
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, str(cm[i][j]), ha='center', va='center', 
                            color='white' if cm[i][j] > cm.max()/2 else 'black')
            plt.savefig("results/prediction_plots/confusion_matrix.png")
            plt.close()
            
            # Identify misclassified examples
            df['correct'] = (y_true == y_pred).astype(int)
            misclassified = df[df['correct'] == 0]
            
            # Save misclassified examples
            if len(misclassified) > 0:
                misclassified_path = "results/prediction_metrics/misclassified.csv"
                misclassified.to_csv(misclassified_path, index=False)
                print(f"\nMisclassified examples saved to {misclassified_path}")
                print(f"Number of misclassified examples: {len(misclassified)} out of {len(df)} ({len(misclassified)/len(df)*100:.2f}%)")
                
        except Exception as e:
            print(f"Warning: Could not compute metrics - {str(e)}")
    else:
        print("\nNo 'label' column found in input data. Skipping evaluation metrics.")
    
    # Save predictions if output path provided
    if output_csv:
        print(f"Saving predictions to {output_csv}...")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"✅ Predictions saved to {output_csv}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate predictions using trained model.')
    parser.add_argument('--model', type=str, default='data/models/fake_review_model.h5',
                        help='Path to trained model file (.h5)')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to model metadata file (.pkl)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions (optional)')
    
    args = parser.parse_args()
    
    # If metadata path not provided, derive it from model path
    if args.metadata is None:
        args.metadata = os.path.splitext(args.model)[0] + "_metadata.pkl"
    
    # Load model and metadata
    model, metadata = load_model_and_metadata(args.model, args.metadata)
    
    # Generate predictions
    predict(model, metadata, args.input, args.output)

if __name__ == "__main__":
    main()