# behavioral_model_trainer.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BehavioralModelTrainer:
    """Train a neural network model for behavioral analysis only"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        self.feature_names = []
        
    def load_behavioral_data(self, data_path):
        """Load the behavioral dataset we created earlier"""
        logger.info(f"Loading behavioral data from {data_path}...")
        
        try:
            df = pd.read_pickle(data_path)
            logger.info(f"✅ Loaded {len(df):,} records with {df['behavioral_label'].nunique()} labels")
            
            # Display dataset info
            label_counts = df['behavioral_label'].value_counts()
            logger.info("📊 Dataset Overview:")
            logger.info(f"   - Total records: {len(df):,}")
            logger.info(f"   - Fake reviews (OR): {label_counts.get('OR', 0):,}")
            logger.info(f"   - Genuine reviews (CG): {label_counts.get('CG', 0):,}")
            
            if 'OR' not in label_counts or label_counts['OR'] == 0:
                logger.warning("⚠️  NO FAKE REVIEWS FOUND IN DATASET!")
                logger.warning("   The model cannot learn to detect fake behavior without examples.")
                logger.warning("   Please check your behavioral labeling thresholds.")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {str(e)}")
            raise
    
    def analyze_class_balance(self, df):
        """Analyze and fix class imbalance"""
        logger.info("🔍 Analyzing class balance...")
        
        label_counts = df['behavioral_label'].value_counts()
        total_samples = len(df)
        
        logger.info("📈 Class Distribution:")
        for label, count in label_counts.items():
            percentage = (count / total_samples) * 100
            logger.info(f"   - {label}: {count:,} ({percentage:.2f}%)")
        
        # Check if we have enough fake samples
        fake_count = label_counts.get('OR', 0)
        genuine_count = label_counts.get('CG', 0)
        
        if fake_count == 0:
            logger.error("❌ CRITICAL: No fake reviews found in dataset!")
            logger.error("   The model cannot learn without positive examples.")
            logger.error("   Please adjust behavioral labeling thresholds to be less strict.")
            raise ValueError("No fake reviews in dataset")
        
        if fake_count < 100:
            logger.warning(f"⚠️  Very few fake reviews: {fake_count}")
            logger.warning("   Consider adjusting thresholds or using data augmentation")
        
        # Calculate imbalance ratio
        imbalance_ratio = genuine_count / max(fake_count, 1)
        logger.info(f"📊 Imbalance Ratio (Genuine:Fake): {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 10:
            logger.warning("⚠️  Severe class imbalance detected!")
            logger.warning("   Will apply class weighting and consider sampling techniques")
        
        return imbalance_ratio
    
    def prepare_behavioral_features(self, df):
        """Prepare features for behavioral model training"""
        logger.info("🔧 Preparing behavioral features...")
        
        # Define behavioral features (NO textual features)
        behavioral_features = [
            # Core behavioral flags
            'high_burst_user', 
            'high_frequency_user', 
            'self_duplicate_flag', 
            'cross_duplicate_flag',
            
            # Behavioral metrics
            'reviews_per_day',
            'burst_ratio', 
            'burst_count',
            'total_reviews',
            'account_age_days',
            'avg_rating',
            'self_similarity_score',
            
            # Temporal features
            'review_year',
            'review_month', 
            'review_hour',
            'day_of_week',
            
            # Helpful votes (social behavior)
            'helpful_count',
            'total_votes', 
            'helpful_ratio'
        ]
        
        # Only use features that exist in the dataframe
        available_features = [f for f in behavioral_features if f in df.columns]
        missing_features = set(behavioral_features) - set(available_features)
        
        if missing_features:
            logger.warning(f"⚠️ Missing features: {missing_features}")
        
        logger.info(f"✅ Using {len(available_features)} behavioral features")
        
        # Create feature matrix
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Prepare labels - ensure we have both classes
        y = df['behavioral_label'].map({'CG': 0, 'OR': 1}).values
        
        # Verify we have both classes
        unique_labels = np.unique(y)
        logger.info(f"🔢 Labels in dataset: {unique_labels}")
        
        if len(unique_labels) == 1:
            logger.error(f"❌ Only one class found: {unique_labels[0]}")
            logger.error("   Cannot train binary classification with one class")
            raise ValueError("Only one class in dataset")
        
        self.feature_names = available_features
        return X, y
    
    def build_model(self, input_shape):
        """Build neural network architecture for behavioral analysis"""
        logger.info(f"🧠 Building neural network with input shape: {input_shape}")
        
        model = Sequential([
            # Input layer
            Dense(128, activation='relu', input_shape=(input_shape,),
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layer 1
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(), 
            Dropout(0.3),
            
            # Hidden layer 2
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        model.summary()
        return model
    
    def calculate_class_weights(self, y):
        """Calculate class weights for imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        logger.info("⚖️ Class weights for imbalance:")
        logger.info(f"   - Class 0 (Genuine): {class_weight_dict[0]:.2f}")
        logger.info(f"   - Class 1 (Fake): {class_weight_dict[1]:.2f}")
        
        return class_weight_dict
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the behavioral model"""
        logger.info("🚀 Training behavioral model...")
        
        # Scale features
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Build model
        input_shape = X_train_scaled.shape[1]
        self.model = self.build_model(input_shape)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(y_train)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=15,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,  # Smaller batch size for better learning
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weights
        )
        
        return X_train_scaled, X_val_scaled
    
    def evaluate_model(self, X_test, y_test, dataset_name="Test"):
        """Comprehensive model evaluation"""
        logger.info(f"📊 Evaluating model on {dataset_name} set...")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle AUC calculation for edge cases
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except ValueError as e:
            logger.warning(f"⚠️  Could not calculate AUC: {e}")
            auc_score = 0.5  # Random classifier
        
        logger.info(f"✅ {dataset_name} Results:")
        logger.info(f"   - Accuracy: {accuracy:.4f}")
        logger.info(f"   - AUC-ROC: {auc_score:.4f}")
        
        # Detailed classification report
        logger.info(f"📋 Classification Report ({dataset_name}):")
        report = classification_report(y_test, y_pred, target_names=['Genuine', 'Fake'])
        logger.info(f"\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"🎯 Confusion Matrix ({dataset_name}):")
        logger.info(f"True Negative (Genuine): {cm[0,0]}")
        logger.info(f"False Positive: {cm[0,1]}") 
        logger.info(f"False Negative: {cm[1,0]}")
        logger.info(f"True Positive (Fake): {cm[1,1]}")
        
        return {
            'accuracy': float(accuracy),
            'auc_roc': float(auc_score),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist(),
            'confusion_matrix': cm.tolist()
        }
    
    def plot_training_history(self, output_dir):
        """Plot training history and metrics"""
        logger.info("📈 Plotting training history...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert history to JSON-serializable format
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        # Accuracy
        axes[0, 0].plot(history_dict['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history_dict['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history_dict['loss'], label='Training Loss')
        axes[0, 1].plot(history_dict['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC
        if 'auc' in history_dict:
            axes[1, 0].plot(history_dict['auc'], label='Training AUC')
            axes[1, 0].plot(history_dict['val_auc'], label='Validation AUC')
            axes[1, 0].set_title('Model AUC')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Precision-Recall
        if 'precision' in history_dict and 'recall' in history_dict:
            axes[1, 1].plot(history_dict['precision'], label='Training Precision')
            axes[1, 1].plot(history_dict['val_precision'], label='Validation Precision')
            axes[1, 1].plot(history_dict['recall'], label='Training Recall')
            axes[1, 1].plot(history_dict['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Precision & Recall')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Training plots saved to {output_dir}/training_history.png")
        
        return history_dict
    
    def plot_feature_importance(self, X_train, output_dir):
        """Plot feature importance based on model weights"""
        logger.info("🔍 Analyzing feature importance...")
        
        # Get weights from first layer
        weights = self.model.layers[0].get_weights()[0]
        feature_importance = np.mean(np.abs(weights), axis=1)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Feature Importance (Absolute Weight)')
        plt.title('Behavioral Feature Importance')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Feature importance plot saved")
        logger.info("📊 Top 5 most important behavioral features:")
        for _, row in importance_df.tail(5).iterrows():
            logger.info(f"   - {row['feature']}: {row['importance']:.4f}")
        
        return importance_df.to_dict('records')
    
    def save_model(self, output_dir, history_dict, results):
        """Save the trained model and metadata"""
        logger.info("💾 Saving model and metadata...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = f"{output_dir}/behavioral_model.h5"
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = f"{output_dir}/behavioral_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature names and metadata
        metadata = {
            'feature_names': self.feature_names,
            'model_type': 'behavioral_analysis',
            'training_date': datetime.now().isoformat(),
            'input_shape': len(self.feature_names),
            'classes': {0: 'genuine', 1: 'fake'},
            'training_history': history_dict
        }
        
        metadata_path = f"{output_dir}/behavioral_model_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save training results as JSON
        results_path = f"{output_dir}/training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Model saved to: {model_path}")
        logger.info(f"✅ Scaler saved to: {scaler_path}")
        logger.info(f"✅ Metadata saved to: {metadata_path}")
        logger.info(f"✅ Training results saved to: {results_path}")
    
    def run_complete_training(self, data_path, test_size=0.2, val_size=0.2):
        """Run complete training pipeline"""
        logger.info("🚀 Starting complete behavioral model training pipeline...")
        
        # Step 1: Load data
        df = self.load_behavioral_data(data_path)
        
        # Step 2: Analyze class balance
        imbalance_ratio = self.analyze_class_balance(df)
        
        # Step 3: Prepare features
        X, y = self.prepare_behavioral_features(df)
        
        # Step 4: Split data with stratification to maintain class balance
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=42, stratify=y_temp
        )
        
        logger.info("📊 Data Split Summary:")
        logger.info(f"   - Training set: {len(X_train):,} samples")
        logger.info(f"   - Validation set: {len(X_val):,} samples") 
        logger.info(f"   - Test set: {len(X_test):,} samples")
        logger.info(f"   - Fake ratio - Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
        
        # Step 5: Train model
        X_train_scaled, X_val_scaled = self.train_model(X_train, y_train, X_val, y_val)
        
        # Step 6: Evaluate model
        train_results = self.evaluate_model(X_train, y_train, "Training")
        val_results = self.evaluate_model(X_val, y_val, "Validation") 
        test_results = self.evaluate_model(X_test, y_test, "Test")
        
        # Step 7: Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"data/models/behavioral_model_{timestamp}"
        
        # Step 8: Generate plots and analysis
        history_dict = self.plot_training_history(output_dir)
        importance_data = self.plot_feature_importance(X_train, output_dir)
        
        # Step 9: Save comprehensive results
        results = {
            'training_metrics': train_results,
            'validation_metrics': val_results,
            'test_metrics': test_results,
            'feature_importance': importance_data,
            'data_statistics': {
                'total_samples': len(df),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test),
                'feature_count': len(self.feature_names),
                'fake_ratio_overall': float(y.mean()),
                'fake_ratio_train': float(y_train.mean()),
                'fake_ratio_test': float(y_test.mean()),
                'imbalance_ratio': float(imbalance_ratio)
            }
        }
        
        # Step 10: Save model and artifacts
        self.save_model(output_dir, history_dict, results)
        
        logger.info(f"✅ Complete training pipeline finished!")
        logger.info(f"📁 All artifacts saved to: {output_dir}")
        
        return results

def main():
    """Main function to run behavioral model training"""
    trainer = BehavioralModelTrainer()
    
    # Path to your behavioral dataset
    data_path = "data/processed/final_labeled_dataset.pkl"
    
    try:
        results = trainer.run_complete_training(data_path)
        
        print("\n" + "="*60)
        print("🎉 BEHAVIORAL MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Print final summary
        test_metrics = results['test_metrics']
        data_stats = results['data_statistics']
        
        print(f"\n📊 FINAL TEST RESULTS:")
        print(f"   - Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   - AUC-ROC: {test_metrics['auc_roc']:.4f}")
        
        print(f"\n📈 DATA STATISTICS:")
        print(f"   - Total samples: {data_stats['total_samples']:,}")
        print(f"   - Behavioral features: {data_stats['feature_count']}")
        print(f"   - Fake review ratio: {data_stats['fake_ratio_overall']:.3f}")
        print(f"   - Imbalance ratio: {data_stats['imbalance_ratio']:.1f}:1")
        
        # Check if model is actually learning
        if test_metrics['auc_roc'] > 0.7:
            print(f"\n✅ Model is learning well! (AUC > 0.7)")
        elif test_metrics['auc_roc'] > 0.6:
            print(f"\n⚠️  Model performance is moderate (AUC > 0.6)")
        else:
            print(f"\n❌ Model performance is poor. Check your data and thresholds.")
        
        print(f"\n✅ Model is ready for integration with your Flask API!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        print(f"\n💡 SUGGESTION: Adjust behavioral labeling thresholds to get more fake reviews")
        print("   In behavioral_analyzer.py, change BEHAVIORAL_THRESHOLDS to be less strict")
        raise

if __name__ == "__main__":
    main()