import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, accuracy_score, 
                            confusion_matrix, roc_auc_score, roc_curve,
                            mean_squared_error, r2_score, precision_recall_curve,
                            average_precision_score, f1_score)

def test_model(model_path, metadata_path, test_csv, output_dir="results/test_results"):
    """Comprehensive evaluation of model on test data."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Load metadata
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Load test data
    print(f"Loading test data from {test_csv}...")
    test_df = pd.read_csv(test_csv)
    
    # Extract features
    all_features = metadata['feature_names']
    X_test = test_df[all_features]
    
    # Map labels
    label_map = metadata['label_map']
    y_test = test_df['label'].map(label_map)
    
    # Apply preprocessing
    print("Preprocessing test data...")
    X_test_scaled = metadata['scaler'].transform(X_test)
    
    # Apply feature selection if available
    if 'selected_indices' in metadata:
        X_test_selected = X_test_scaled[:, metadata['selected_indices']]
        selected_features = metadata['selected_features']
    else:
        X_test_selected = X_test_scaled
        selected_features = all_features
    
    # Generate predictions
    print("Generating predictions...")
    y_proba = model.predict(X_test_selected)
    
    # Ensure probabilities are properly shaped
    if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
        y_proba = y_proba[:, 1]
    else:
        y_proba = y_proba.flatten()
    
    y_pred = (y_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_proba)
    
    # Print metrics
    print("\n=== Test Set Evaluation ===")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save metrics to file
    with open(f"{output_dir}/test_metrics.txt", "w") as f:
        f.write("Test Set Metrics:\n")
        f.write(f"Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"AUC-ROC: {auc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R² Score: {r2:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Test Set)')
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Test Set)')
    plt.legend(loc="lower left")
    plt.savefig(f"{output_dir}/precision_recall_curve.png")
    plt.close()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Test Set)')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1], ['Genuine', 'Fake'])
    plt.yticks([0, 1], ['Genuine', 'Fake'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center', 
                    color='white' if cm[i][j] > cm.max()/2 else 'black')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    
    # Add predictions to dataframe
    test_df['predicted_probability'] = y_proba
    test_df['predicted_label'] = y_pred
    test_df['correct_prediction'] = (y_pred == y_test).astype(int)
    
    # Save predictions with detailed analysis
    test_df.to_csv(f"{output_dir}/test_predictions.csv", index=False)
    
    # Find misclassified examples
    misclassified = test_df[test_df['correct_prediction'] == 0].copy()
    misclassified.to_csv(f"{output_dir}/misclassified.csv", index=False)
    
    # Get false positives and false negatives
    false_positives = test_df[(test_df['label'] == 'OR') & (test_df['predicted_label'] == 1)]
    false_negatives = test_df[(test_df['label'] == 'CG') & (test_df['predicted_label'] == 0)]
    
    # Save separate files
    false_positives.to_csv(f"{output_dir}/false_positives.csv", index=False)
    false_negatives.to_csv(f"{output_dir}/false_negatives.csv", index=False)
    
    print(f"\n✅ Test evaluation completed. Results saved to {output_dir}")
    print(f"Misclassified examples: {len(misclassified)} out of {len(test_df)} ({len(misclassified)/len(test_df)*100:.2f}%)")
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'avg_precision': avg_precision,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the trained model on test data.')
    parser.add_argument('--model', type=str, default='data/models/fake_review_model.h5',
                        help='Path to trained model file (.h5)')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to model metadata file (.pkl)')
    parser.add_argument('--test-data', type=str, default='data/analysis/test_analysis.csv',
                        help='Path to test CSV file')
    parser.add_argument('--output-dir', type=str, default='results/test_results',
                        help='Directory to save test results')
    
    args = parser.parse_args()
    
    # If metadata path not provided, derive it from model path
    if args.metadata is None:
        args.metadata = os.path.splitext(args.model)[0] + "_metadata.pkl"
    
    # Run test evaluation
    test_model(args.model, args.metadata, args.test_data, args.output_dir)