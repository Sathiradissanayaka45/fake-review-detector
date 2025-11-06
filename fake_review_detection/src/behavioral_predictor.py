# behavioral_predictor.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BehavioralPredictor:
    """Behavioral analysis model for prediction"""
    
    def __init__(self, model_dir=None):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = []
        
        if model_dir:
            self.load_model(model_dir)
    
    def load_model(self, model_dir):
        """Load trained behavioral model"""
        logger.info(f"Loading behavioral model from {model_dir}...")
        
        try:
            # Load model
            self.model = load_model(f"{model_dir}/behavioral_model.h5")
            
            # Load scaler
            with open(f"{model_dir}/behavioral_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load metadata
            with open(f"{model_dir}/behavioral_model_metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)
            
            self.feature_names = self.metadata['feature_names']
            
            logger.info(f"✅ Behavioral model loaded successfully")
            logger.info(f"   - Features: {len(self.feature_names)}")
            logger.info(f"   - Input shape: {self.metadata['input_shape']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load behavioral model: {str(e)}")
            raise
    
    def extract_behavioral_features(self, user_data):
        """
        Extract behavioral features from user data
        user_data should be a dictionary with user behavior metrics
        """
        features = {}
        
        # Core behavioral flags
        features['high_burst_user'] = user_data.get('high_burst_user', 0)
        features['high_frequency_user'] = user_data.get('high_frequency_user', 0)
        features['self_duplicate_flag'] = user_data.get('self_duplicate_flag', 0)
        features['cross_duplicate_flag'] = user_data.get('cross_duplicate_flag', 0)
        
        # Behavioral metrics
        features['reviews_per_day'] = user_data.get('reviews_per_day', 0)
        features['burst_ratio'] = user_data.get('burst_ratio', 0)
        features['burst_count'] = user_data.get('burst_count', 0)
        features['total_reviews'] = user_data.get('total_reviews', 0)
        features['account_age_days'] = user_data.get('account_age_days', 1)  # Avoid division by zero
        features['avg_rating'] = user_data.get('avg_rating', 3.0)
        features['self_similarity_score'] = user_data.get('self_similarity_score', 0)
        
        # Temporal features (use current time if not provided)
        current_time = datetime.now()
        features['review_year'] = user_data.get('review_year', current_time.year)
        features['review_month'] = user_data.get('review_month', current_time.month)
        features['review_hour'] = user_data.get('review_hour', current_time.hour)
        features['day_of_week'] = user_data.get('day_of_week', current_time.weekday())
        
        # Social behavior
        features['helpful_count'] = user_data.get('helpful_count', 0)
        features['total_votes'] = user_data.get('total_votes', 0)
        features['helpful_ratio'] = user_data.get('helpful_ratio', 0)
        
        # Ensure all expected features are present
        for feature in self.feature_names:
            if feature not in features:
                features[feature] = 0  # Default value for missing features
        
        return features
    
    def predict_behavioral(self, user_data):
        """
        Predict if user behavior indicates fake reviews
        
        Args:
            user_data: Dictionary with user behavioral metrics
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Extract features
            features = self.extract_behavioral_features(user_data)
            
            # Create feature vector in correct order
            feature_vector = [features[feature] for feature in self.feature_names]
            X = np.array([feature_vector])
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict
            probability = self.model.predict(X_scaled, verbose=0)[0][0]
            prediction = 1 if probability >= 0.5 else 0
            
            # Generate behavioral flags
            behavioral_flags = self._generate_behavioral_flags(features, probability)
            
            return {
                'prediction': 'fake' if prediction == 1 else 'genuine',
                'probability': float(probability),
                'behavioral_flags': behavioral_flags,
                'user_metrics': self._summarize_user_metrics(features),
                'confidence': self._calculate_confidence(probability)
            }
            
        except Exception as e:
            logger.error(f"❌ Behavioral prediction failed: {str(e)}")
            return {
                'prediction': 'error',
                'probability': 0.5,
                'behavioral_flags': [],
                'user_metrics': {},
                'error': str(e)
            }
    
    def _generate_behavioral_flags(self, features, probability):
        """Generate specific behavioral flags based on features"""
        flags = []
        
        # Burst behavior flags
        if features.get('high_burst_user', 0) == 1:
            flags.append({
                'type': 'review_burst_detected',
                'severity': 'high',
                'description': 'User posts multiple reviews in very short time windows'
            })
        
        if features.get('burst_ratio', 0) > 0.3:
            flags.append({
                'type': 'frequent_bursting', 
                'severity': 'medium',
                'description': f"High burst ratio: {features['burst_ratio']:.2f}"
            })
        
        # Frequency flags
        if features.get('high_frequency_user', 0) == 1:
            flags.append({
                'type': 'unusual_review_frequency',
                'severity': 'high', 
                'description': f"Extremely high review frequency: {features.get('reviews_per_day', 0):.2f} reviews/day"
            })
        elif features.get('reviews_per_day', 0) > 3:
            flags.append({
                'type': 'high_review_frequency',
                'severity': 'medium',
                'description': f"High review frequency: {features['reviews_per_day']:.2f} reviews/day"
            })
        
        # Duplication flags
        if features.get('self_duplicate_flag', 0) == 1:
            flags.append({
                'type': 'copy_paste_behavior',
                'severity': 'high',
                'description': 'User copies and pastes their own reviews'
            })
        
        if features.get('cross_duplicate_flag', 0) == 1:
            flags.append({
                'type': 'cross_user_duplication', 
                'severity': 'high',
                'description': 'User posts reviews identical to other users'
            })
        
        # Account pattern flags
        if features.get('total_reviews', 0) > 50 and features.get('account_age_days', 1) < 30:
            flags.append({
                'type': 'suspicious_account_growth',
                'severity': 'medium',
                'description': 'Many reviews in very short account lifetime'
            })
        
        # Rating pattern flags
        avg_rating = features.get('avg_rating', 3.0)
        if avg_rating > 4.8 or avg_rating < 1.2:
            flags.append({
                'type': 'extreme_rating_bias',
                'severity': 'medium', 
                'description': f'Extreme average rating: {avg_rating:.1f} stars'
            })
        
        return flags
    
    def _summarize_user_metrics(self, features):
        """Summarize key user metrics"""
        return {
            'reviews_per_day': round(features.get('reviews_per_day', 0), 2),
            'total_reviews': features.get('total_reviews', 0),
            'account_age_days': features.get('account_age_days', 0),
            'burst_ratio': round(features.get('burst_ratio', 0), 3),
            'avg_rating': round(features.get('avg_rating', 0), 1),
            'helpful_ratio': round(features.get('helpful_ratio', 0), 3)
        }
    
    def _calculate_confidence(self, probability):
        """Calculate confidence level based on probability"""
        distance_from_decision = abs(probability - 0.5)
        
        if distance_from_decision > 0.3:
            return 'high'
        elif distance_from_decision > 0.15:
            return 'medium'
        else:
            return 'low'

# Utility function to find latest model
def get_latest_behavioral_model():
    """Find the most recently trained behavioral model"""
    models_dir = "data/models"
    behavioral_dirs = [d for d in os.listdir(models_dir) if d.startswith('behavioral_model_')]
    
    if not behavioral_dirs:
        raise FileNotFoundError("No behavioral models found. Please train a model first.")
    
    # Sort by timestamp (newest first)
    latest_dir = sorted(behavioral_dirs, reverse=True)[0]
    return os.path.join(models_dir, latest_dir)

# Singleton instance for easy access
_behavioral_predictor = None

def get_behavioral_predictor():
    """Get singleton behavioral predictor instance"""
    global _behavioral_predictor
    if _behavioral_predictor is None:
        model_dir = get_latest_behavioral_model()
        _behavioral_predictor = BehavioralPredictor(model_dir)
    return _behavioral_predictor

if __name__ == "__main__":
    # Test the predictor
    predictor = BehavioralPredictor(get_latest_behavioral_model())
    
    # Test with sample user data
    test_user = {
        'high_burst_user': 1,
        'high_frequency_user': 0,
        'self_duplicate_flag': 1,
        'cross_duplicate_flag': 0,
        'reviews_per_day': 2.5,
        'burst_ratio': 0.4,
        'total_reviews': 15,
        'account_age_days': 10,
        'avg_rating': 4.8
    }
    
    result = predictor.predict_behavioral(test_user)
    print("🧪 Behavioral Prediction Test:")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.3f}")
    print(f"Confidence: {result['confidence']}")
    print("Flags:", [f["type"] for f in result['behavioral_flags']])