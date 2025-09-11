import os
import numpy as np
import pandas as pd
from utils import flag_robotic_reviews

def analyze_dataset(input_csv, output_csv):
    """Perform readability and sentiment analysis."""
    print(f"\nAnalyzing {input_csv}...")
    
    try:
        # Load data
        df = pd.read_csv(input_csv)
        
        # Data validation
        required_columns = ['cleaned_text', 'readability_score', 'word_count', 'rating', 'sentiment_score']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        print("\nOriginal data shape:", df.shape)
        
        # Readability analysis
        print("\nPerforming readability analysis...")
        df['robotic_flag'] = df.apply(
            lambda row: flag_robotic_reviews(
                row['cleaned_text'],
                row['readability_score'],
                row['word_count']
            ), axis=1
        )
        
        # Sentiment-rating mismatch analysis
        print("Performing sentiment-rating mismatch analysis...")
        df['normalized_rating'] = (df['rating'] - 3) / 2  # Convert 1-5 scale to -1 to 1
        df['mismatch_score'] = np.abs(
            df['normalized_rating'] - df['sentiment_score']
        )
        df['mismatch_flag'] = (df['mismatch_score'] > 0.5).astype(int)
        
        # Save results
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        
        # Print summary
        print("\nAnalysis summary:")
        print(f"- {df['robotic_flag'].sum()} robotic reviews flagged")
        print(f"- {df['mismatch_flag'].sum()} sentiment mismatches flagged")
        print(f"✅ Analysis saved to {output_csv}")
        
    except Exception as e:
        print(f"\n❌ Error in data analysis: {str(e)}")
        raise

if __name__ == "__main__":
    analyze_dataset(
        input_csv="data/features/train_features.csv",
        output_csv="data/analysis/train_analysis.csv"
    )