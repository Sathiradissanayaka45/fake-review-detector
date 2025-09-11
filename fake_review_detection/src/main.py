from data_split import split_dataset
from data_preprocessing import preprocess_dataset
from feature_extraction import extract_features
from analyze_data import analyze_dataset
from train_model import train_model
import os

def run_pipeline():
    print("🚀 Starting pipeline...")
    
    # Create all directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/features", exist_ok=True)
    os.makedirs("data/analysis", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("results/validation_plots", exist_ok=True)
    
    try:
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
        
        # 3. Feature extraction
        print("\n=== STEP 3: Feature Extraction ===")
        for split in ['train', 'val', 'test']:
            extract_features(
                input_csv=f"data/processed/{split}_processed.csv",
                output_csv=f"data/features/{split}_features.csv"
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
        train_model(
            train_csv="data/analysis/train_analysis.csv",
            val_csv="data/analysis/val_analysis.csv",
            model_output="data/models/fake_review_model.pkl"
        )
        
        print("\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()