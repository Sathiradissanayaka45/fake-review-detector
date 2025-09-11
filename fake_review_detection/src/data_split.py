import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(input_csv, output_dir):
    """Split dataset into train/val/test with stratified sampling."""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the raw data
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Verify label distribution
    print("\nOriginal label distribution:")
    print(df['label'].value_counts(normalize=True))
    
    # First split: 80% train, 20% temp (for val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    # Second split: 50% val, 50% test (which is 10% of original each)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['label']
    )
    
    # Verify splits
    print("\nSplit sizes:")
    print(f"Training set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    print("\nLabel distribution in splits:")
    print("Training:")
    print(train_df['label'].value_counts(normalize=True))
    print("\nValidation:")
    print(val_df['label'].value_counts(normalize=True))
    print("\nTest:")
    print(test_df['label'].value_counts(normalize=True))
    
    # Save the splits
    train_path = os.path.join(output_dir, "train_raw.csv")
    val_path = os.path.join(output_dir, "val_raw.csv")
    test_path = os.path.join(output_dir, "test_raw.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✅ Datasets saved to:")
    print(f"- Training: {train_path}")
    print(f"- Validation: {val_path}")
    print(f"- Testing: {test_path}")

if __name__ == "__main__":
    split_dataset(
        input_csv="data/raw/raw_reviews.csv",
        output_dir="data/splits"
    )