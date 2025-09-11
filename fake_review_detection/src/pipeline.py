from data_split import split_dataset
from data_preprocessing import preprocess_dataset
from feature_extraction import extract_features
from enhanced_feature_extraction import EnhancedFeatureExtractor
from analyze_data import analyze_dataset
from train_model import train_model
import os
import tensorflow as tf
import argparse

def run_pipeline(run_full=False, use_bert=True, bert_batch_size=32):
    """
    Run the complete data processing and model training pipeline.
    
    Args:
        run_full: If True, run the full pipeline from data splitting to model training.
                 If False, only run the model training step.
        use_bert: If True, use BERT embeddings for enhanced feature extraction.
        bert_batch_size: Batch size for BERT processing.
    """
    print("🚀 Starting enhanced pipeline...")
    print(f"BERT embeddings: {'✅ Enabled' if use_bert else '❌ Disabled'}")
    
    # Print TensorFlow version for debugging
    print(f"TensorFlow version: {tf.__version__}")
    
    # Create all directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/features", exist_ok=True)
    os.makedirs("data/analysis", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("results/validation_plots", exist_ok=True)
    
    try:
        if run_full:
            # 1. Split data
            print("\n=== STEP 1: Splitting Data ===")
            split_dataset(
                input_csv="data/raw/raw_reviews.csv",
                output_dir="data/splits"
            )
            
            # 2. Preprocess all splits
            print("\n=== STEP 2: Preprocessing ===")
            for split in ['train', 'val', 'test']:
                preprocess_dataset(
                    input_csv=f"data/splits/{split}_raw.csv",
                    output_csv=f"data/processed/{split}_processed.csv"
                )
            
            # 3. Enhanced Feature extraction with BERT
            print("\n=== STEP 3: Enhanced Feature Extraction ===")
            
            # Initialize enhanced feature extractor
            feature_extractor = EnhancedFeatureExtractor(
                use_bert=use_bert,
                bert_batch_size=bert_batch_size
            )
            
            # Extract features for training data (fits TF-IDF vectorizer)
            print("\n🔸 Processing training data...")
            feature_extractor.extract_features(
                input_csv="data/processed/train_processed.csv",
                output_csv="data/features/train_features.csv",
                is_training=True
            )
            
            # Extract features for validation and test data (uses fitted vectorizer)
            for split in ['val', 'test']:
                print(f"\n🔸 Processing {split} data...")
                feature_extractor.extract_features(
                    input_csv=f"data/processed/{split}_processed.csv",
                    output_csv=f"data/features/{split}_features.csv",
                    is_training=False
                )
            
            # 4. Analysis
            print("\n=== STEP 4: Data Analysis ===")
            for split in ['train', 'val', 'test']:
                analyze_dataset(
                    input_csv=f"data/features/{split}_features.csv",
                    output_csv=f"data/analysis/{split}_analysis.csv"
                )
        
            # 5. Model training (only uses train and val)
            print("\n=== STEP 5: Model Training ===")
            model, metrics = train_model(
                train_csv="data/analysis/train_analysis.csv",
                val_csv="data/analysis/val_analysis.csv",
                model_output="data/models/fake_review_model.h5"
            )
        
        print("\n✅ Enhanced pipeline completed successfully!")
        print(f"BERT embeddings: {'✅ Used' if use_bert else '❌ Not used'}")
        print(f"Final model accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Final model AUC-ROC: {metrics['roc_auc']:.4f}")
        print(f"Final model R² Score: {metrics['r2_score']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the enhanced fake review detection pipeline.')
    parser.add_argument('--run_full', action='store_true', 
                      help='Run the full pipeline including data splitting and preprocessing')
    parser.add_argument('--no-bert', action='store_true',
                      help='Disable BERT embeddings (use TF-IDF only)')
    parser.add_argument('--bert-batch-size', type=int, default=32,
                      help='Batch size for BERT processing (default: 32)')
    
    args = parser.parse_args()
    
    # Run pipeline with arguments
    run_pipeline(
        run_full=args.run_full,
        use_bert=not args.no_bert,  # BERT enabled by default
        bert_batch_size=args.bert_batch_size
    )