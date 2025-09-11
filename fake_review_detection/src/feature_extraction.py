import os
import re
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import extract_linguistic_features

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def extract_features(input_csv, output_csv):
    """Extract all features including TF-IDF."""
    print(f"\nExtracting features from {input_csv}...")
    
    try:
        # Load data
        df = pd.read_csv(input_csv)
        
        # Verify data
        print("\nLabel distribution:")
        print(df['label'].value_counts())
        
        # Data validation
        if 'cleaned_text' not in df.columns:
            raise ValueError("Input CSV must contain 'cleaned_text' column")
        if 'label' not in df.columns:
            raise ValueError("Input CSV must contain 'label' column")
        
        # Clean and prepare data
        df['cleaned_text'] = df['cleaned_text'].astype(str).fillna('')
        
        # Extract linguistic features with progress bar
        print("\nExtracting linguistic features...")
        tqdm.pandas(desc="Progress")
        features_df = df["cleaned_text"].progress_apply(
            lambda x: pd.Series(extract_linguistic_features(x))
        )
        df = pd.concat([df, features_df], axis=1)

        # TF-IDF processing
        print("\nProcessing TF-IDF features...")
        if "train" in input_csv:
            # Fit and save vectorizer for training data
            tfidf = TfidfVectorizer(
                max_features=200,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=5,
                max_df=0.95
            )
            tfidf_features = tfidf.fit_transform(df["cleaned_text"])
            
            # Ensure models directory exists
            os.makedirs("data/models", exist_ok=True)
            
            # Save vectorizer
            with open("data/models/tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(tfidf, f)
            print("✅ TF-IDF vectorizer saved")
        else:
            # Load vectorizer for validation/test data
            try:
                with open("data/models/tfidf_vectorizer.pkl", "rb") as f:
                    tfidf = pickle.load(f)
                tfidf_features = tfidf.transform(df["cleaned_text"])
            except FileNotFoundError:
                raise FileNotFoundError(
                    "TF-IDF vectorizer not found. Train the model first to create it."
                )

        # Create TF-IDF feature columns
        tfidf_cols = pd.DataFrame(
            tfidf_features.toarray(), 
            columns=[f'tfidf_{feat}' for feat in tfidf.get_feature_names_out()]
        )

        # Combine all features
        final_df = pd.concat([df, tfidf_cols], axis=1)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        # Save final features
        final_df.to_csv(output_csv, index=False)
        print(f"\n✅ Features saved to {output_csv}")
        print(f"Total features extracted: {len(final_df.columns)}")
        print(f"Feature names: {list(final_df.columns)}")
        
    except Exception as e:
        print(f"\n❌ Error in feature extraction: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    extract_features(
        input_csv="data/processed/train_processed.csv",
        output_csv="data/features/train_features.csv"
    )