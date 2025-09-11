import pandas as pd
import os
from tqdm import tqdm
from utils import clean_text

def preprocess_dataset(input_csv, output_csv):
    """Process dataset with progress tracking."""
    print(f"Preprocessing {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Data validation
    print("\nData validation:")
    print("Unique labels:", df['label'].unique())
    print("Label counts:\n", df['label'].value_counts())
    
    # Clean text with progress bar
    tqdm.pandas(desc="Cleaning text")
    df["cleaned_text"] = df["text_"].progress_apply(clean_text)
    
    # Save results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Cleaned data saved to {output_csv}")

if __name__ == "__main__":
    # Example usage (normally called from pipeline.py)
    preprocess_dataset(
        input_csv="data/splits/train_raw.csv",
        output_csv="data/processed/train_processed.csv"
    )