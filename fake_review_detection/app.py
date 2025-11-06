import json
import time  # Add this import for correct sleep functionality
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mysqldb import MySQL
import pandas as pd
import numpy as np
import spacy
from textstat import flesch_reading_ease
from textblob import TextBlob
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os
import re
import datetime
from werkzeug.utils import secure_filename
import warnings
from datetime import datetime, timedelta 
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import shap
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from src.behavioral_predictor import get_behavioral_predictor, get_latest_behavioral_model

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# MySQL Configuration
app.config['MYSQL_HOST'] = os.environ.get('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.environ.get('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.environ.get('MYSQL_PASSWORD', 'Dilshara224#')
app.config['MYSQL_DB'] = os.environ.get('MYSQL_DB', 'review_analysis')
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# Initialize MySQL
mysql = MySQL(app)

# Add connection retry logic
def init_mysql_connection():
    max_retries = 5
    retry_delay = 5  # seconds
    
    with app.app_context():  # Ensure application context
        for attempt in range(max_retries):
            try:
                # Test connection
                cur = mysql.connection.cursor()
                cur.execute("SELECT 1")
                cur.close()
                logger.info("✅ MySQL connection established successfully")
                return True
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: MySQL connection failed - {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)  # Use correct time.sleep
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("❌ Failed to establish MySQL connection after multiple attempts")
                    raise

# Initialize MySQL connection
try:
    init_mysql_connection()
except Exception as e:
    logger.error(f"Failed to initialize MySQL: {str(e)}")
    raise
# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to load SpaCy model: {str(e)}")
    raise

# Load the trained model, metadata, and TF-IDF vectorizer
try:
    model = load_model("data/models/fake_review_model.h5")
    with open("data/models/fake_review_model_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    scaler = metadata['scaler']
    feature_cols = metadata['feature_names']
    label_map = metadata['label_map']
    with open("data/models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    background_data = np.load("data/models/fake_review_model_background.npy")
    explainer = shap.DeepExplainer(model, background_data)
    logger.info("✅ Model, metadata, TF-IDF, and SHAP explainer loaded")
except Exception as e:
    logger.error(f"❌ Error loading: {str(e)}")
    raise


class RuleBasedBehavioralAnalyzer:
    """Simple, rule-based behavioral analysis that actually works"""
    
    def __init__(self):
        self.feature_names = ['reviews_per_day', 'account_age_days', 'total_reviews', 'avg_rating']
        self.metadata = {
            'model_type': 'rule_based_behavioral',
            'training_date': '2024-01-01'
        }
    
    def predict_behavioral(self, user_data):
        """Simple, logical behavioral analysis"""
        try:
            # Extract basic metrics
            reviews_per_day = float(user_data.get('reviews_per_day', 0))
            account_age_days = int(user_data.get('account_age_days', 1))
            total_reviews = int(user_data.get('total_reviews', 0))
            avg_rating = float(user_data.get('avg_rating', 3.0))
            
            # Calculate fake probability based on CLEAR rules
            fake_probability = self._calculate_fake_probability(
                reviews_per_day, account_age_days, total_reviews, avg_rating
            )
            
            # Generate behavioral flags
            behavioral_flags = self._generate_behavioral_flags(
                reviews_per_day, account_age_days, total_reviews, avg_rating
            )
            
            # Determine prediction (unchanged)
            if fake_probability >= 0.7:
                prediction = "fake"
                confidence = "high"
            elif fake_probability >= 0.5:
                prediction = "fake"
                confidence = "medium"
            elif fake_probability >= 0.3:
                prediction = "genuine"
                confidence = "medium"
            else:
                prediction = "genuine"
                confidence = "high"
            
            # FIXED: Set probability to the probability of the PREDICTED class
            # (Matches textual analysis format)
            if prediction == "fake":
                prob = fake_probability
            else:
                prob = 1 - fake_probability
            
            # Ensure bounded (min 0.05, max 0.95) for the predicted prob
            prob = max(0.05, min(0.95, prob))
            
            return {
                "prediction": prediction,
                "probability": round(prob, 3),
                "confidence": confidence,
                "behavioral_flags": behavioral_flags,
                "user_metrics": {
                    'reviews_per_day': reviews_per_day,
                    'account_age_days': account_age_days,
                    'total_reviews': total_reviews,
                    'avg_rating': avg_rating,
                    'burst_ratio': 0,
                    'helpful_ratio': 0
                },
                "calculated_metrics": {}
            }
            
        except Exception as e:
            logger.error(f"Error in behavioral analysis: {str(e)}")
            return self._get_fallback_prediction()
    
    def _calculate_fake_probability(self, reviews_per_day, account_age, total_reviews, avg_rating):
        """Calculate fake probability based on logical rules"""
        fake_score = 0.0
        
        # RULE 1: Review frequency (Most important)
        if reviews_per_day > 10.0:  # Obvious bot
            fake_score += 0.8
        elif reviews_per_day > 5.0:  # Very suspicious
            fake_score += 0.6
        elif reviews_per_day > 2.0:  # Suspicious
            fake_score += 0.4
        elif reviews_per_day > 1.0:  # Moderate
            fake_score += 0.2
        elif reviews_per_day < 0.01:  # Too inactive (could be fake too)
            fake_score += 0.1
        
        # RULE 2: Account age patterns
        if account_age < 7 and total_reviews > 10:  # New account, many reviews
            fake_score += 0.7
        elif account_age < 30 and total_reviews > 20:  # Young account, high activity
            fake_score += 0.5
        elif account_age < 7:  # Very new account
            fake_score += 0.3
        
        # RULE 3: Rating patterns
        if avg_rating > 4.9 or avg_rating < 1.1:  # Extreme ratings only
            fake_score += 0.4
        elif avg_rating == 5.0 or avg_rating == 1.0:  # Only perfect or worst ratings
            fake_score += 0.3
        
        # RULE 4: Total reviews patterns
        if total_reviews > 100 and account_age < 60:  # Too many reviews too quickly
            fake_score += 0.6
        elif total_reviews > 50 and account_age < 30:
            fake_score += 0.4
        
        # PREVENT EXTREME PROBABILITIES (never exactly 0.0 or 1.0)
        final_probability = min(0.95, fake_score)
        
        # Apply reasonable bounds
        if final_probability < 0.05:  # Very genuine
            return 0.05  # Minimum 5% fake probability
        elif final_probability > 0.95:  # Very fake
            return 0.95  # Maximum 95% fake probability
        else:
            return round(final_probability, 3)  # Keep original with rounding
    
    def _generate_behavioral_flags(self, reviews_per_day, account_age, total_reviews, avg_rating):
        """Generate clear behavioral flags"""
        flags = []
        
        # Review frequency flags
        if reviews_per_day > 5.0:
            flags.append({
                "type": "extremely_high_frequency",
                "severity": "high",
                "description": f"Extremely high review frequency: {reviews_per_day:.2f} reviews/day"
            })
        elif reviews_per_day > 2.0:
            flags.append({
                "type": "high_review_frequency", 
                "severity": "medium",
                "description": f"High review frequency: {reviews_per_day:.2f} reviews/day"
            })
        
        # Account age flags
        if account_age < 7 and total_reviews > 5:
            flags.append({
                "type": "suspicious_new_account",
                "severity": "high",
                "description": f"New account ({account_age} days) with {total_reviews} reviews"
            })
        elif account_age < 30 and total_reviews > 20:
            flags.append({
                "type": "high_activity_young_account",
                "severity": "medium",
                "description": f"Young account ({account_age} days) with high activity"
            })
        
        # Rating pattern flags
        if avg_rating > 4.9:
            flags.append({
                "type": "suspiciously_positive",
                "severity": "medium",
                "description": f"Extremely positive rating pattern: {avg_rating:.1f} stars average"
            })
        elif avg_rating < 1.1:
            flags.append({
                "type": "suspiciously_negative",
                "severity": "medium", 
                "description": f"Extremely negative rating pattern: {avg_rating:.1f} stars average"
            })
        
        return flags
    
    def _get_fallback_prediction(self):
        """Fallback when analysis fails"""
        return {
            "prediction": "genuine",
            "probability": 0.5,
            "confidence": "low",
            "behavioral_flags": [],
            "user_metrics": {},
            "calculated_metrics": {},
            "error": "Analysis failed, using fallback"
        }

# Initialize the simple behavioral analyzer
behavioral_predictor = RuleBasedBehavioralAnalyzer()
logger.info("✅ Rule-based behavioral analyzer initialized successfully")
    
# Text processing constants
CONTRACTIONS = {
    "don't": "do not", "can't": "cannot", "won't": "will not",
    "it's": "it is", "i'm": "i am", "they're": "they are",
    "you're": "you are", "we're": "we are", "he's": "he is",
    "she's": "she is", "that's": "that is", "what's": "what is"
}

def clean_text(text):
    if not isinstance(text, str):
        logger.warning("Input text is not a string, returning empty string")
        return ""
    
    text = text.lower()
    for key, value in CONTRACTIONS.items():
        text = text.replace(key, value)
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        doc = nlp(text)
        cleaned_text = " ".join([
            token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct and len(token.text) > 1
        ])
        return cleaned_text or ""
    except Exception as e:
        logger.error(f"Error in clean_text: {str(e)}")
        return ""

def get_sentiment(text):
    """Get sentiment from original text, not cleaned text."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    
    try:
        # Use original text for sentiment analysis, just basic preprocessing
        # Expand contractions but keep important words like "not", "is", etc.
        processed_text = text.lower()
        for key, value in CONTRACTIONS.items():
            processed_text = processed_text.replace(key, value)
        
        analysis = TextBlob(processed_text)
        sentiment = analysis.sentiment.polarity
        return max(-1.0, min(1.0, sentiment))
    except Exception as e:
        logger.error(f"Error in get_sentiment: {str(e)}")
        return 0.0

# Add flag_robotic_reviews (from original pipeline)
def flag_robotic_reviews(text, readability_score, word_count):
    """Flag suspicious reviews based on readability."""
    if not isinstance(text, str):
        return 0
    if readability_score > 80 and word_count < 15:
        return 1  # Too simple
    elif readability_score < 30 and word_count > 50:
        return 1  # Overly complex
    return 0

def extract_linguistic_features(text, rating=None, original_text=None):
    """
    Extract comprehensive linguistic features that match the model's training features.
    Includes TF-IDF and robotic_flag/mismatch_flag.
    
    Args:
        text: Cleaned text for most features
        rating: Optional rating for mismatch analysis
        original_text: Original text for sentiment analysis
    """
    logger.debug("Extracting features for text: %s", text[:50] + "..." if len(text) > 50 else text)
    
    if not isinstance(text, str) or not text.strip():
        text = ""
    
    try:
        doc = nlp(text)
    except Exception as e:
        logger.error(f"Error processing text with SpaCy: {str(e)}")
        doc = nlp("")
    
    try:
        readability = flesch_reading_ease(text)
    except Exception as e:
        logger.error(f"Error in flesch_reading_ease: {str(e)}")
        readability = 60.0
    
    word_count = len([t for t in doc if not t.is_punct])
    
    # Use original text for sentiment analysis if provided, otherwise use cleaned text
    sentiment_text = original_text if original_text else text
    
    features = {
        'word_count': word_count,
        'sentence_count': len(list(doc.sents)) or 1,
        'avg_word_length': np.mean([len(t.text) for t in doc]) if doc else 0,
        'readability_score': readability,
        'sentiment_score': get_sentiment(sentiment_text),
        'exclamation_count': text.count('!'),
        'first_person_pronouns': sum(1 for t in doc if t.lemma_ in ['i', 'me', 'my', 'mine', 'myself']),
        'superlative_words': sum(1 for t in doc if t.tag_ in ['JJS', 'RBS']),
        'adj_ratio': 0,
        'adv_ratio': 0,
        'noun_ratio': 0,
        'verb_ratio': 0,
        # Add missing features from analyze_dataset
        'robotic_flag': flag_robotic_reviews(text, readability, word_count),
        'mismatch_flag': 0  # Will be set below if rating is provided
    }
    
    pos_counts = {'ADJ': 0, 'ADV': 0, 'NOUN': 0, 'VERB': 0}
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
    
    total_words = max(1, word_count)
    features['adj_ratio'] = pos_counts['ADJ'] / total_words
    features['adv_ratio'] = pos_counts['ADV'] / total_words
    features['noun_ratio'] = pos_counts['NOUN'] / total_words
    features['verb_ratio'] = pos_counts['VERB'] / total_words
    
    if rating is not None:
        try:
            rating = float(rating)
            if 1 <= rating <= 5:
                features['rating'] = rating
                features['normalized_rating'] = (rating - 3) / 2
                features['mismatch_score'] = np.abs(
                    features['normalized_rating'] - features['sentiment_score']
                )
                features['mismatch_flag'] = 1 if features['mismatch_score'] > 0.5 else 0
            else:
                logger.warning("Rating %s out of valid range (1-5), ignoring rating features", rating)
                rating = None
        except (ValueError, TypeError):
            logger.warning("Invalid rating value: %s, ignoring rating features", rating)
            rating = None
    
    # Compute TF-IDF features
    try:
        tfidf_features = tfidf_vectorizer.transform([text]).toarray()[0]
        tfidf_cols = [f'tfidf_{feat}' for feat in tfidf_vectorizer.get_feature_names_out()]
        for col, value in zip(tfidf_cols, tfidf_features):
            features[col] = value
    except Exception as e:
        logger.error(f"Error in TF-IDF transformation: {str(e)}")
        tfidf_cols = [f'tfidf_{feat}' for feat in tfidf_vectorizer.get_feature_names_out()]
        for col in tfidf_cols:
            features[col] = 0.0
    
    return features

def store_review_in_db(text, rating, result):
    """Store review and analysis results in MySQL database."""
    try:
        cur = mysql.connection.cursor()
        
        # Insert review data
        cur.execute("""
            INSERT INTO reviews1 
            (review_text, rating, prediction, probability, word_count, 
             sentiment_score, readability_score, exclamation_count,
             first_person_pronouns, superlative_words, adj_ratio,
             adv_ratio, noun_ratio, verb_ratio, mismatch_score, robotic_flag, created_at) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            text[:1000],  # Truncate text to avoid length issues
            rating if rating is not None else None,
            result['prediction'],
            result['probability'],
            result['features']['word_count'],
            result['features']['sentiment_score'],
            result['features']['readability_score'],
            result['features']['exclamation_count'],
            result['features']['first_person_pronouns'],
            result['features']['superlative_words'],
            result['features']['adj_ratio'],
            result['features']['adv_ratio'],
            result['features']['noun_ratio'],
            result['features']['verb_ratio'],
            result['features'].get('mismatch_score', 0),
            result['features'].get('robotic_flag', 0),
            datetime.now()
        ))
        
        review_id = cur.lastrowid
        
        # Insert warning flags
        for flag in result['explanation']['warning_flags']:
            cur.execute("""
                INSERT INTO warning_flags1 (review_id, flag_type) VALUES (%s, %s)
            """, (review_id, flag['type']))
        
        mysql.connection.commit()
        cur.close()
        logger.info("Successfully stored review in database, review_id: %s", review_id)
        return True
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        mysql.connection.rollback()
        return False

# Define research-based reason mappings
fake_indicators = {
    #Research (e.g., Ott et al., 2011) shows this is common in deceptive text to mimic enthusiasm or criticism without genuine experience.
    'superlative_words': "Overuse of superlatives and extreme language: Fake reviews frequently use words like 'best,' 'worst,' 'amazing,' or 'terrible' to exaggerate impact.",
    #Studies (e.g., Mukherjee et al., 2013) indicate fakes often have this inconsistency because generators focus on ratings over coherent content.
    'mismatch_score': "Mismatched sentiment and rating: The text's emotional tone (e.g., positive words) doesn't align with the star rating (e.g., 5 stars but negative sentiment).",
    #Research (e.g., Feng et al., 2012) links this to non-human patterns.
    'robotic_flag': "Unnatural readability (Too simple or too complex): Fake reviews may be overly simplistic (high readability_score with short word_count) from templates or bots, or overly complex (low readability_score with long word_count) from AI-generated text.",
    #Behavioral studies (e.g., Levitan et al., 2018) show deceivers use less first-person language to distance themselves.
    'first_person_pronouns': "Lack of personal pronouns: Genuine experiences use 'I,' 'me,' or 'my,' but fakes avoid them to stay generic.",
    #Psycholinguistic research (e.g., from LIWC tools) associates this with manipulative emphasis.
    'exclamation_count': "Excessive exclamations or punctuation: Fakes overuse '!' to fake excitement.",
    'sentiment_score': "Extreme sentiment polarity: Fakes tend to be overly positive or negative without nuance. Sentiment analysis research (e.g., using TextBlob or VADER) finds genuine reviews have more balanced or neutral tones.",
    #Review spam detection (e.g., Crawford et al., 2015) notes fakes deviate from natural review lengths.
    'word_count': "Unusual length or structure: Often shorter (low word_count) for quick fakes or longer with filler (high sentence_count but low verb_ratio/noun_ratio).",
    'adj_ratio': "Unbalanced adjective use: High adj_ratio indicates over-descriptive language common in fakes to sound convincing.",
    'adv_ratio': "Unbalanced adverb use: High adv_ratio suggests unnatural emphasis, a hallmark of manipulative text.",
    #Content analysis (e.g., Harris, 2012) shows deceivers skip specifics.
    'noun_ratio': "Lack of specific details: Low noun_ratio means fewer concrete nouns (e.g., product names), which fakes avoid to prevent contradictions.",
    'verb_ratio': "Unusual verb usage: Low verb_ratio can indicate lack of action-oriented narrative, common in generic fakes."
}

tfidf_fake_reason = "Repetitive or generic phrasing: Common in bot-generated reviews, detected via n-grams or word patterns. TF-IDF-based studies (e.g., Jindal & Liu, 2008) show fakes reuse phrases like 'great product' across items."

genuine_indicators = {
    #(e.g., Vásquez, 2014).
    'superlative_words': "Balanced language without exaggeration: Moderate superlatives and exclamations. Genuine reviews avoid hype, as per content analysis",
    #Studies (e.g., Hu et al., 2012) show genuine reviews have low mismatch_scores because users report honestly.
    'mismatch_score': "Consistent sentiment and rating: The text's tone matches the rating (e.g., positive words with 4-5 stars).",
    #Psycholinguistic research (e.g., Pennebaker et al., 2007) links this to authentic writing.
    'robotic_flag': "Natural readability: Falls in a moderate range (readability_score 30-80), reflecting everyday language.",
    #Deception detection (e.g., Newman et al., 2003) finds truth-tellers use more self-references.
    'first_person_pronouns': "High use of personal pronouns: Frequent 'I,' 'me,' or 'my' indicate personal experience.",
    'exclamation_count': "Appropriate exclamations: Balanced punctuation for authentic expression.",
    #Sentiment analysis research (e.g., Pang & Lee, 2008) shows genuine reviews express nuanced opinions.
    'sentiment_score': "Nuanced sentiment: Mild polarity (sentiment_score near 0 for balanced views).",
    #Review length studies (e.g., Mudambi & Schuff, 2010) show genuines have natural lengths.
    'word_count': "Natural length and structure: Typical word_count (15-50 words) with varied sentences.",
    'adj_ratio': "Balanced adjective use: Appropriate descriptiveness without overdoing.",
    'adv_ratio': "Balanced adverb use: Natural emphasis in authentic language.",
    #Content analysis (e.g., Harris, 2012) shows genuines include specific details.
    'noun_ratio': "Specific and descriptive content: Higher noun_ratio (product details) and verb_ratio (actions like 'tried,' 'used').",
    'verb_ratio': "Natural verb usage: Balanced action words for coherent narratives."
}
#Duplicate detection (e.g., Ott et al., 2013) shows genuines are original.  
tfidf_genuine_reason = "Unique phrasing: Low similarity to templates, captured by diverse TF-IDF scores."

def combine_predictions(textual_result, behavioral_result, interaction_analysis=None):
    """
    CORRECTED: Properly combine textual and behavioral predictions
    """
    # Get predictions and probabilities
    textual_pred = textual_result['prediction']
    textual_prob = textual_result['probability']
    behavioral_pred = behavioral_result['prediction']
    behavioral_prob = behavioral_result['probability']
    
    # DEBUG: Log incoming data
    logger.debug(f"COMBINE_PREDICTIONS - Textual: {textual_pred} (prob: {textual_prob})")
    logger.debug(f"COMBINE_PREDICTIONS - Behavioral: {behavioral_pred} (prob: {behavioral_prob})")
    
    # CORRECTED: Proper probability interpretation
    if behavioral_pred == 'fake':
        behavioral_fake_prob = behavioral_prob
    else:  # behavioral_pred == 'genuine'
        behavioral_fake_prob = 1 - behavioral_prob
    
    # For textual analysis
    if textual_pred == 'fake':
        textual_fake_prob = textual_prob
    else:
        textual_fake_prob = 1 - textual_prob
    
    logger.debug(f"Textual fake probability: {textual_fake_prob}")
    logger.debug(f"Behavioral fake probability: {behavioral_fake_prob}")
    
    # Give more weight to textual analysis (it's more reliable)
    textual_weight = 0.7  # Increased weight for textual
    behavioral_weight = 0.3  # Reduced weight for behavioral
    
    # Calculate combined probability
    combined_fake_prob = (
        textual_fake_prob * textual_weight + 
        behavioral_fake_prob * behavioral_weight
    )
    
    # Apply interaction trust adjustment (but less aggressively)
    interaction_trust = interaction_analysis.get('calculated_trust_score', 100) if interaction_analysis else 100
    distrust_factor = (100 - interaction_trust) / 100
    distrust_adjustment = distrust_factor * 0.1  # Reduced from 0.2 to 0.1
    
    adjusted_combined_prob = min(0.99, combined_fake_prob + distrust_adjustment)
    
    # More reasonable thresholds
    if adjusted_combined_prob >= 0.7:  # Lowered from 0.8
        final_prediction = 'fake'
        confidence_level = 'high'
        reason = "Strong evidence of fake review"
    elif adjusted_combined_prob >= 0.55:  # Lowered from 0.6
        final_prediction = 'fake' 
        confidence_level = 'medium'
        reason = "Moderate evidence indicating fake activity"
    elif adjusted_combined_prob >= 0.45:  # Wider genuine range
        final_prediction = 'suspicious'
        confidence_level = 'medium'
        reason = "Mixed signals detected"
    else:
        final_prediction = 'genuine'
        confidence_level = 'high' 
        reason = "Consistent genuine signals"
    
    trust_score = int((1 - adjusted_combined_prob) * 100)
    
    # Generate comprehensive flags
    comprehensive_flags = generate_comprehensive_flags(textual_result, behavioral_result)
    
    # Add interaction flags but be less aggressive
    if interaction_analysis:
        # Only add high severity flags
        high_severity_flags = [flag for flag in interaction_analysis.get('behavioral_flags', []) 
                              if flag.get('severity') == 'high']
        comprehensive_flags.extend(high_severity_flags)
    
    combined_explanation = generate_combined_explanation(
        textual_result, behavioral_result, final_prediction, adjusted_combined_prob
    )
    
    return {
        'final_prediction': final_prediction,
        'combined_score': round(adjusted_combined_prob, 3),
        'trust_score': trust_score,
        'confidence': confidence_level,
        'reason': reason,
        'comprehensive_flags': comprehensive_flags,
        'combined_explanation': combined_explanation,
        'interaction_analysis': interaction_analysis,
        'textual_analysis': {
            'prediction': textual_pred,
            'probability': round(textual_prob, 3),
            'confidence': textual_result['confidence'],
            'key_factors': textual_result['explanation']['key_factors'],
            'warning_flags': textual_result['explanation']['warning_flags'],
            'detailed_reasons': textual_result['explanation'].get('detailed_reasons', [])
        },
        'behavioral_analysis': {
            'prediction': behavioral_pred,
            'probability': round(behavioral_prob, 3),  # ← CORRECTED: Use original probability
            'confidence': behavioral_result['confidence'],
            'behavioral_flags': behavioral_result.get('behavioral_flags', []),
            'user_metrics': behavioral_result.get('user_metrics', {}),
            'calculated_metrics': behavioral_result.get('calculated_metrics', {})
        }
    }
    
def calculate_confidence_weight(probability):
    """Calculate confidence weight based on probability distance from 0.5"""
    distance_from_decision = abs(probability - 0.5)
    
    if distance_from_decision > 0.4:
        return 3.0  # Very high confidence
    elif distance_from_decision > 0.3:
        return 2.0  # High confidence
    elif distance_from_decision > 0.2:
        return 1.5  # Medium confidence
    else:
        return 1.0  # Low confidence

def generate_comprehensive_flags(textual_result, behavioral_result):
    """Generate comprehensive flags combining both analyses"""
    flags = []
    
    # Textual flags
    for flag in textual_result['explanation']['warning_flags']:
        flags.append({
            'type': f'textual_{flag["type"]}',
            'severity': 'medium',
            'source': 'textual_analysis',
            'description': flag['description']
        })
    
    # Behavioral flags
    for flag in behavioral_result['behavioral_flags']:
        flags.append({
            'type': f'behavioral_{flag["type"]}',
            'severity': flag['severity'],
            'source': 'behavioral_analysis',
            'description': flag['description']
        })
    
    # Combined risk assessment
    textual_risk = 1 if textual_result['prediction'] == 'fake' else 0
    behavioral_risk = 1 if behavioral_result['prediction'] == 'fake' else 0
    
    if textual_risk == 1 and behavioral_risk == 1:
        flags.append({
            'type': 'high_risk_combined',
            'severity': 'high',
            'source': 'combined_analysis',
            'description': 'Both textual and behavioral analysis indicate fake activity'
        })
    elif textual_risk == 1 or behavioral_risk == 1:
        flags.append({
            'type': 'medium_risk_single',
            'severity': 'medium',
            'source': 'combined_analysis',
            'description': 'One analysis method indicates potential fake activity'
        })
    
    return flags
  
def predict_review(text, rating=None):
    try:
        logger.debug("Starting prediction for text: %s", text[:50] + "..." if len(text) > 50 else text)
        
        cleaned_text = clean_text(text)
        if not cleaned_text:
            logger.warning("Cleaned text is empty for input: %s", text[:50])
        
        features = extract_linguistic_features(cleaned_text, rating, original_text=text)
        logger.debug("Extracted features: %s", list(features.keys()))
        
        features_df = pd.DataFrame([features])
        
        missing_cols = set(feature_cols) - set(features_df.columns)
        if missing_cols:
            logger.warning("Missing features: %s, setting to 0", missing_cols)
            for col in missing_cols:
                features_df[col] = 0.0
        
        features_df = features_df[feature_cols]
        logger.debug("Feature DataFrame shape: %s", features_df.shape)
        
        features_scaled = scaler.transform(features_df)
        
        prediction_proba = model.predict(features_scaled, verbose=0)[0][0]
        
        prediction = 1 if prediction_proba >= 0.5 else 0
        
        # Compute SHAP
        shap_values = explainer.shap_values(features_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]  # Positive class (fake)
        else:
            shap_values = shap_values[0]
        
        # Convert features to a dictionary with ensured float values
        feature_values = {col: float(features.get(col, 0.0)) for col in feature_cols}
        
        # Explanation
        explanation = {
            "key_factors": {
                "readability": {"value": feature_values.get('readability_score', 0.0), "description": "Measures how easy the text is to read"},
                "sentiment": {"value": feature_values.get('sentiment_score', 0.0), "description": "Positive/negative sentiment strength (-1 to 1)"},
                "word_count": {"value": feature_values.get('word_count', 0.0), "description": "Number of words in the review"},
                "first_person_usage": {"value": feature_values.get('first_person_pronouns', 0.0), "description": "Count of 'I', 'me', 'my' pronouns"},
                "superlatives": {"value": feature_values.get('superlative_words', 0.0), "description": "Count of superlative words (best, worst, etc.)"}
            },
            "warning_flags": [],
            "detailed_reasons": []
        }
        
        if rating is not None and 'mismatch_score' in features:
            if features['mismatch_score'] > 0.5:
                explanation['warning_flags'].append({
                    "type": "sentiment_rating_mismatch",
                    "description": "The review sentiment doesn't match the rating"
                })
        
        if features['robotic_flag']:
            if features['readability_score'] > 80 and features['word_count'] < 15:
                explanation['warning_flags'].append({
                    "type": "overly_simple_language",
                    "description": "The review is unusually simple for its length"
                })
            elif features['readability_score'] < 30 and features['word_count'] > 50:
                explanation['warning_flags'].append({
                    "type": "overly_complex_language",
                    "description": "The review is unusually complex for its length"
                })
        
        if features['first_person_pronouns'] > 5:
            explanation['warning_flags'].append({
                "type": "excessive_first_person_usage",
                "description": "High usage of first-person pronouns"
            })
        
        if features['superlative_words'] > 3:
            explanation['warning_flags'].append({
                "type": "excessive_superlatives",
                "description": "High number of superlative words"
            })
# Enhanced detailed reasons from SHAP
        detailed_reasons = []
        tfidf_contributed = False
        threshold = 0.01
        
        if prediction == 1:  # Fake: Focus on positive SHAP
            for feat, shap_val in sorted(zip(feature_cols, shap_values), key=lambda x: x[1], reverse=True):
                if shap_val > threshold:
                    if feat in fake_indicators:
                        detailed_reasons.append({
                            "reason": fake_indicators[feat],
                            "feature": feat,
                            "value": float(feature_values.get(feat, 0.0)),
                            "shap_impact": float(shap_val)
                        })
                    elif feat.startswith('tfidf_') and not tfidf_contributed:
                        detailed_reasons.append({
                            "reason": tfidf_fake_reason,
                            "feature": feat,
                            "value": float(feature_values.get(feat, 0.0)),
                            "shap_impact": float(shap_val)
                        })
                        tfidf_contributed = True
        else:  # Genuine: Negative SHAP
            for feat, shap_val in sorted(zip(feature_cols, shap_values), key=lambda x: x[1]):
                if shap_val < -threshold:
                    if feat in genuine_indicators:
                        detailed_reasons.append({
                            "reason": genuine_indicators[feat],
                            "feature": feat,
                            "value": float(feature_values.get(feat, 0.0)),
                            "shap_impact": float(shap_val)
                        })
                    elif feat.startswith('tfidf_') and not tfidf_contributed:
                        detailed_reasons.append({
                            "reason": tfidf_genuine_reason,
                            "feature": feat,
                            "value": float(feature_values.get(feat, 0.0)),
                            "shap_impact": float(shap_val)
                        })
                        tfidf_contributed = True
        
        explanation['detailed_reasons'] = detailed_reasons
        
        result = {
            "prediction": "fake" if prediction == 1 else "genuine",
            "probability": float(prediction_proba if prediction == 1 else 1 - prediction_proba),
            "confidence": "high" if abs(prediction_proba - 0.5) > 0.3 else "medium" if abs(prediction_proba - 0.5) > 0.15 else "low",
            "explanation": explanation,
            "features": feature_values
        }
        
        if not store_review_in_db(text, rating, result):
            logger.warning("Failed to store review in database, continuing with prediction")
        
        logger.info("Prediction completed: %s (probability: %.4f)", result['prediction'], result['probability'])
        return result
    
    except Exception as e:
        logger.error(f"Error in predict_review: {str(e)}")
        return {"error": str(e), "details": "Failed to process review"}

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for single review prediction."""
    try:
        data = request.get_json()
        logger.debug("Received request data: %s", data)
        
        if not data or 'text' not in data:
            logger.warning("Missing 'text' field in request")
            return jsonify({
                "error": "Invalid request",
                "message": "Missing required field 'text'"
            }), 400
        
        text = data['text']
        rating = data.get('rating')
        
        if not isinstance(text, str) or not text.strip():
            logger.warning("Invalid or empty text: %s", text)
            return jsonify({
                "error": "Invalid input",
                "message": "Text must be a non-empty string"
            }), 400
        
        # Validate rating if provided
        if rating is not None:
            try:
                rating = float(rating)
                if not (1 <= rating <= 5):
                    logger.warning("Rating %s out of valid range (1-5), setting to None", rating)
                    rating = None
            except (ValueError, TypeError):
                logger.warning("Invalid rating value: %s, setting to None", rating)
                rating = None
        
        result = predict_review(text, rating)
        
        if 'error' in result:
            logger.error("Prediction failed: %s", result['error'])
            return jsonify(result), 500
            
        return jsonify({
            "success": True,
            "result": result
        })
    
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {str(e)}")
        return jsonify({
            "error": "Server error",
            "message": str(e)
        }), 500
    
    except Exception as e:
        return jsonify({
            "error": "Server error",
            "message": str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch predictions from CSV file."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                "error": "No file uploaded",
                "message": "Please upload a CSV file"
            }), 400
            
        file = request.files['file']
        
        # Validate file
        if file.filename == '':
            return jsonify({
                "error": "Empty filename",
                "message": "No file selected"
            }), 400
            
        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                "error": "Invalid file type",
                "message": "Only CSV files are supported"
            }), 400
        
        # Save uploaded file
        os.makedirs('uploads', exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        # Read CSV file
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return jsonify({
                "error": "Invalid CSV file",
                "message": str(e)
            }), 400
        
        # Validate required columns
        if 'text' not in df.columns:
            return jsonify({
                "error": "Missing column",
                "message": "CSV must contain a 'text' column"
            }), 400
        
        # Process each review
        results = []
        for _, row in df.iterrows():
            text = row['text']
            rating = row.get('rating')
            
            if pd.isna(text) or not str(text).strip():
                results.append({
                    "error": "Empty text",
                    "input": row.to_dict()
                })
                continue
                
            result = predict_review(str(text), float(rating) if not pd.isna(rating) else None)
            results.append(result)
            
        # Clean up - remove uploaded file
        try:
            os.remove(filepath)
        except:
            pass
            
        return jsonify({
            "success": True,
            "processed": len(results),
            "results": results
        })
        
    except Exception as e:
        return jsonify({
            "error": "Batch processing failed",
            "message": str(e)
        }), 500

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Endpoint to get analytics data for dashboard."""
    try:
        cur = mysql.connection.cursor()
        
        # Get prediction distribution
        cur.execute("""
            SELECT 
                prediction,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM reviews1), 2) as percentage
            FROM reviews1
            GROUP BY prediction
            ORDER BY count DESC
        """)
        prediction_stats = cur.fetchall()
        
        # Get sentiment distribution
        cur.execute("""
            SELECT 
                CASE 
                    WHEN sentiment_score > 0.3 THEN 'Positive'
                    WHEN sentiment_score < -0.3 THEN 'Negative'
                    ELSE 'Neutral'
                END as sentiment,
                COUNT(*) as count,
                ROUND(AVG(probability), 2) as avg_probability
            FROM reviews1
            GROUP BY sentiment
            ORDER BY count DESC
        """)
        sentiment_stats = cur.fetchall()
        
        # Get daily prediction trend
        cur.execute("""
            SELECT 
                DATE(created_at) as date,
                prediction,
                COUNT(*) as count
            FROM reviews1
            WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
            GROUP BY DATE(created_at), prediction
            ORDER BY date
        """)
        daily_trend = cur.fetchall()
        
        # Get recent reviews
        cur.execute("""
            SELECT 
                review_id,
                review_text,
                rating,
                prediction,
                probability,
                created_at
            FROM reviews1
            ORDER BY created_at DESC
            LIMIT 5
        """)
        recent_reviews = cur.fetchall()
        
        cur.close()
        
        # Convert to proper JSON serializable format
        return jsonify({
            "success": True,
            "analytics": {
                "prediction_stats": [
                    {"prediction": item["prediction"], "count": item["count"], "percentage": float(item["percentage"])}
                    for item in prediction_stats
                ],
                "sentiment_stats": [
                    {"sentiment": item["sentiment"], "count": item["count"], "avg_probability": float(item["avg_probability"])}
                    for item in sentiment_stats
                ],
                "daily_trend": [
                    {"date": item["date"].strftime("%Y-%m-%d"), "prediction": item["prediction"], "count": item["count"]}
                    for item in daily_trend
                ],
                "recent_reviews": [
                    {
                        "review_text": item["review_text"],
                        "rating": item["rating"],
                        "prediction": item["prediction"],
                        "probability": float(item["probability"])
                    }
                    for item in recent_reviews
                ]
            }
        })
        
    except Exception as e:
        print(f"Database error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Database connection failed",
            "message": str(e)
        }), 500
@app.route('/demo')
def demo_page():
    """Serve the enhanced demo page."""
    from flask import render_template_string
    try:
        with open('templates/enhanced_demo.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return "<h1>Demo page not found</h1><p>Please make sure the templates/enhanced_demo.html file exists.</p>", 404

@app.route('/review/<int:review_id>', methods=['GET'])
def get_review_details(review_id):
    """Endpoint to get detailed analysis for a specific review."""
    try:
        cur = mysql.connection.cursor()
        
        # Get review details
        cur.execute("""
            SELECT * FROM reviews1 WHERE id = %s
        """, (review_id,))
        review = cur.fetchone()
        
        if not review:
            return jsonify({
                "error": "Not found",
                "message": "Review not found"
            }), 404
        
        # Get warning flags for this review
        cur.execute("""
            SELECT flag_type FROM warning_flags1 WHERE review_id = %s
        """, (review_id,))
        flags = [row['flag_type'] for row in cur.fetchall()]
        
        cur.close()
        
        return jsonify({
            "success": True,
            "review": review,
            "warning_flags": flags
        })
        
    except Exception as e:
        return jsonify({
            "error": "Database error",
            "message": str(e)
        }), 500


@app.route('/analytics/fake_review_types', methods=['GET'])
def get_fake_review_types():
    """Get statistics on types of fake reviews detected"""
    try:
        cur = mysql.connection.cursor()
        
        # Get all fake reviews with their features
        cur.execute("""
            SELECT 
                r.*,
                GROUP_CONCAT(wf.flag_type) as warning_flags
            FROM reviews1 r
            LEFT JOIN warning_flags1 wf ON r.review_id = wf.review_id
            WHERE r.prediction = 'fake'
            GROUP BY r.review_id
        """)
        fake_reviews = cur.fetchall()
        
        if not fake_reviews:
            return jsonify({
                "message": "No fake reviews found in database",
                "analysis": {}
            })
        
        # Analyze patterns in fake reviews
        analysis = {
            "total_fake_reviews": len(fake_reviews),
            "common_patterns": {
                "by_readability": analyze_by_readability(fake_reviews),
                "by_sentiment": analyze_by_sentiment(fake_reviews),
                "by_length": analyze_by_length(fake_reviews),
                "by_rating": analyze_by_rating(fake_reviews),
                "common_flags": analyze_warning_flags(fake_reviews),
                "time_trends": analyze_time_trends(fake_reviews)
            },
            "most_common_types": get_most_common_types(fake_reviews)
        }
        
        cur.close()
        return jsonify({
            "success": True,
            "analysis": analysis
        })
        
    except Exception as e:
        logger.error(f"Error in fake review analysis: {str(e)}")
        return jsonify({
            "error": "Analysis failed",
            "message": str(e)
        }), 500

@app.route('/analyze_user_behavior', methods=['POST'])
def analyze_user_behavior():
    """Simplified behavioral analysis with only 5 input fields"""
    try:
        data = request.get_json()
        logger.debug("Received simplified behavioral analysis request: %s", data)
        
        if not data:
            return jsonify({
                "error": "Invalid request",
                "message": "Missing request data"
            }), 400
        
        # Extract and validate required fields
        user_id = data.get('user_id')
        reviews_per_day = data.get('reviews_per_day')
        total_reviews = data.get('total_reviews')
        account_age_days = data.get('account_age_days')
        
        # Validate required fields
        if not all([user_id, reviews_per_day is not None, total_reviews is not None, account_age_days is not None]):
            return jsonify({
                "error": "Missing required fields",
                "message": "All fields are required: user_id, reviews_per_day, total_reviews, account_age_days"
            }), 400
        
        # Convert to appropriate types
        try:
            reviews_per_day = float(reviews_per_day)
            total_reviews = int(total_reviews)
            account_age_days = int(account_age_days)
            avg_rating = float(data.get('avg_rating', 3.0)) if data.get('avg_rating') else 3.0
        except (ValueError, TypeError) as e:
            return jsonify({
                "error": "Invalid data types",
                "message": "reviews_per_day and avg_rating must be numbers, total_reviews and account_age_days must be integers"
            }), 400
        
        # Validate positive values
        if reviews_per_day < 0 or total_reviews < 0 or account_age_days < 0:
            return jsonify({
                "error": "Invalid values",
                "message": "All values must be positive numbers"
            }), 400
        
        # Calculate derived behavioral features
        calculated_metrics = calculate_derived_behavioral_features({
            'reviews_per_day': reviews_per_day,
            'total_reviews': total_reviews,
            'account_age_days': account_age_days,
            'avg_rating': avg_rating
        })
        
        # Create user data with calculated values
        user_data = {
            'user_id': user_id,
            'reviews_per_day': reviews_per_day,
            'total_reviews': total_reviews,
            'account_age_days': account_age_days,
            'avg_rating': avg_rating,
            **calculated_metrics  # Include all calculated metrics
        }
        
        if behavioral_predictor is None:
            return jsonify({
                "error": "Service unavailable",
                "message": "Behavioral analysis model not loaded"
            }), 503
        
        # Perform behavioral analysis
        behavioral_result = behavioral_predictor.predict_behavioral(user_data)
        
        # Add calculated metrics to result for display
        behavioral_result['calculated_metrics'] = calculated_metrics
        
        # Store the analysis in database
        store_behavioral_analysis(user_data, behavioral_result)
        
        return jsonify({
            "success": True,
            "behavioral_analysis": behavioral_result
        })
        
    except Exception as e:
        logger.error(f"Error in simplified behavioral analysis: {str(e)}")
        return jsonify({
            "error": "Behavioral analysis failed",
            "message": str(e)
        }), 500

def calculate_derived_behavioral_features(basic_data):
    """Calculate all behavioral features based on the 5 basic inputs"""
    
    reviews_per_day = basic_data['reviews_per_day']
    total_reviews = basic_data['total_reviews']
    account_age_days = basic_data['account_age_days']
    avg_rating = basic_data['avg_rating']
    
    calculated = {}
    
    # Calculate burst ratio (reviews per day relative to account age)
    if account_age_days > 0:
        expected_reviews_per_day = total_reviews / account_age_days
        calculated['burst_ratio'] = min(1.0, reviews_per_day / max(1, expected_reviews_per_day))
    else:
        calculated['burst_ratio'] = 1.0
    
    # Determine if high burst user
    calculated['high_burst_user'] = 1 if calculated['burst_ratio'] > 0.7 else 0
    
    # Calculate burst count (estimated based on reviews per day)
    calculated['burst_count'] = min(10, int(reviews_per_day * 0.5))
    
    # Calculate self-similarity based on review frequency
    calculated['self_similarity_score'] = min(0.8, reviews_per_day * 0.1)
    
    # Determine high frequency user
    calculated['high_frequency_user'] = 1 if reviews_per_day > 5 else 0
    
    # Calculate helpful ratio based on behavior patterns
    if reviews_per_day < 2:
        calculated['helpful_ratio'] = 0.7  # Normal users have higher helpful ratios
    elif reviews_per_day > 10:
        calculated['helpful_ratio'] = 0.2  # High-frequency users have lower helpful ratios
    else:
        calculated['helpful_ratio'] = 0.5  # Moderate users
    
    # Set helpful count and total votes based on total reviews
    calculated['helpful_count'] = int(total_reviews * calculated['helpful_ratio'])
    calculated['total_votes'] = total_reviews * 2  # Assume 2 votes per review on average
    
    # Set duplication flags based on behavior patterns
    calculated['self_duplicate_flag'] = 1 if reviews_per_day > 8 else 0
    calculated['cross_duplicate_flag'] = 1 if calculated['burst_ratio'] > 0.8 else 0
    
    # Add temporal features (current time)
    current_time = datetime.now()
    calculated['review_year'] = current_time.year
    calculated['review_month'] = current_time.month
    calculated['review_hour'] = current_time.hour
    calculated['day_of_week'] = current_time.weekday()
    
    return calculated

# Add this function to analyze behavioral metrics
def analyze_behavioral_metrics(behavior_metrics):
    """LESS AGGRESSIVE behavioral analysis with enhanced metrics"""
    behavioral_flags = []
    trust_score_deduction = 0
    
    completion_time = behavior_metrics.get('formCompletionTime', 0) / 1000
    
    if completion_time < 2:
        behavioral_flags.append({
            "type": "suspiciously_fast_completion",
            "severity": "medium",
            "description": f"Form completed quickly ({completion_time:.1f}s)",
            "impact": "Possible automated behavior"
        })
        trust_score_deduction += 10
    
    copy_paste_events = behavior_metrics.get('copyPasteEvents', 0)
    if copy_paste_events > 5:
        behavioral_flags.append({
            "type": "excessive_copy_paste", 
            "severity": "low",
            "description": f"Multiple copy-paste events ({copy_paste_events})",
            "impact": "Possible template use"
        })
        trust_score_deduction += 5
    
    field_navigation = behavior_metrics.get('fieldNavigation', [])
    if len(field_navigation) > 0:
        logical_fields = ['review_text', 'user_id', 'reviews_per_day', 'total_reviews', 'account_age_days']
        navigation_matches = sum(1 for i, field in enumerate(field_navigation[:len(logical_fields)]) 
                                 if i < len(logical_fields) and field == logical_fields[i])
        
        if navigation_matches < len(field_navigation) * 0.3:
            behavioral_flags.append({
                "type": "illogical_field_navigation",
                "severity": "low",
                "description": "Field navigation pattern appears somewhat random",
                "impact": "Minor concern"
            })
            trust_score_deduction += 2
    
    # NEW: Typing pattern entropy (variance)
    typing_speed = behavior_metrics.get('typingSpeed', [])
    if len(typing_speed) > 5:  # Need enough data points
        typing_variance = np.var(typing_speed)
        if typing_variance < 50:  # Low variance threshold (robotic typing)
            behavioral_flags.append({
                "type": "low_typing_entropy",
                "severity": "medium",
                "description": f"Low typing pattern variance ({typing_variance:.1f}ms) - indicates robotic input",
                "impact": "Possible bot or scripted behavior"
            })
            trust_score_deduction += 15  # Moderate deduction
    
    # NEW: Mouse movement pattern (if tracked)
    mouse_intervals = behavior_metrics.get('mouseMovementIntervals', [])
    if len(mouse_intervals) > 5:
        mouse_variance = np.var(mouse_intervals)
        if mouse_variance < 100:  # Lower threshold for mouse (more variable naturally)
            behavioral_flags.append({
                "type": "robotic_mouse_pattern",
                "severity": "medium",
                "description": f"Low mouse movement variance ({mouse_variance:.1f}ms) - unnatural patterns",
                "impact": "Possible automated mouse simulation"
            })
            trust_score_deduction += 15
    
    return {
        "behavioral_flags": behavioral_flags,
        "trust_score_deduction": min(trust_score_deduction, 50),  # Increased cap for new features
        "calculated_trust_score": max(50, 100 - trust_score_deduction)  # Lower min if more aggressive
    }
    
def store_behavioral_analysis(user_data, result):
    """Store behavioral analysis results in database"""
    try:
        cur = mysql.connection.cursor()
        
        # Calculate trust score
        trust_score = int((1 - result['probability']) * 100)
        
        cur.execute("""
            INSERT INTO behavioral_analysis 
            (user_id, reviews_per_day, total_reviews, account_age_days, avg_rating,
             prediction, probability, confidence_level, trust_score, behavioral_flags, 
             calculated_metrics, created_at) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_data['user_id'],
            user_data['reviews_per_day'],
            user_data['total_reviews'],
            user_data['account_age_days'],
            user_data['avg_rating'],
            result['prediction'],
            result['probability'],
            result['confidence'],
            trust_score,
            json.dumps(result.get('behavioral_flags', [])),
            json.dumps(result.get('calculated_metrics', {})),
            datetime.now()  # FIXED: Use datetime.now() directly
        ))
        
        analysis_id = cur.lastrowid
        mysql.connection.commit()
        cur.close()
        logger.info("Successfully stored behavioral analysis, analysis_id: %s", analysis_id)
        return True
        
    except Exception as e:
        logger.error(f"Database error in behavioral analysis: {str(e)}")
        mysql.connection.rollback()
        return False

    
@app.route('/analyze_comprehensive', methods=['POST'])
def analyze_comprehensive():
    """Combine textual and behavioral analysis for comprehensive review assessment"""
    try:
        data = request.get_json()
        logger.debug("Received comprehensive analysis request: %s", data)
        
        # Validate required fields
        if not data or 'text' not in data:
            return jsonify({
                "error": "Invalid request", 
                "message": "Missing required field 'text'"
            }), 400
        
        if 'user_data' not in data:
            return jsonify({
                "error": "Invalid request",
                "message": "Missing required field 'user_data'"
            }), 400
        
        text = data['text']
        rating = data.get('rating')
        user_data = data['user_data']
        behavior_metrics = data.get('behavior_metrics', {})
        
        # Validate user_data fields
        required_user_fields = ['user_id', 'reviews_per_day', 'total_reviews', 'account_age_days', 'avg_rating']
        for field in required_user_fields:
            if field not in user_data:
                return jsonify({
                    "error": "Missing user data",
                    "message": f"Missing required user field: {field}"
                }), 400
        
        # Perform textual analysis (unchanged)
        textual_result = predict_review(text, rating)
        if 'error' in textual_result:
            return jsonify({
                "error": "Textual analysis failed",
                "message": textual_result['error']
            }), 500
        
        # Perform behavioral analysis
        if behavioral_predictor is None:
            behavioral_result = {
                "prediction": "unknown",
                "probability": 0.5,
                "behavioral_flags": [],
                "user_metrics": {},
                "confidence": "low",
                "error": "Behavioral model not available"
            }
        else:
            behavioral_result = behavioral_predictor.predict_behavioral(user_data)
        
        # Analyze interaction behavior metrics
        interaction_analysis = analyze_behavioral_metrics(behavior_metrics)
        
        # Add interaction flags to behavioral result
        behavioral_result['behavioral_flags'] = behavioral_result.get('behavioral_flags', []) + interaction_analysis['behavioral_flags']
        
        # Combine results intelligently (with interaction trust score)
        combined_result = combine_predictions(textual_result, behavioral_result, interaction_analysis)
        
        # Store comprehensive analysis in database
        store_comprehensive_analysis(text, rating, combined_result, user_data.get('user_id'))
        
        return jsonify({
            "success": True,
            "comprehensive_analysis": combined_result
        })
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {str(e)}")
        return jsonify({
            "error": "Comprehensive analysis failed", 
            "message": str(e)
        }), 500
        
# def combine_predictions(textual_result, behavioral_result):
#     """
#     Intelligently combine textual and behavioral predictions
#     Uses weighted scoring based on confidence and evidence
#     """
#     textual_pred = textual_result['prediction']  # 'fake' or 'genuine'
#     textual_prob = textual_result['probability']
#     behavioral_pred = behavioral_result['prediction']
#     behavioral_prob = behavioral_result['probability']
    
#     # Convert to numerical scores (fake = 1, genuine = 0)
#     textual_score = 1 if textual_pred == 'fake' else 0
#     behavioral_score = 1 if behavioral_pred == 'fake' else 0
    
#     # Calculate confidence weights
#     textual_confidence = calculate_confidence_weight(textual_prob)
#     behavioral_confidence = calculate_confidence_weight(behavioral_prob)
    
#     # Weighted combination
#     combined_score = (textual_score * textual_confidence + behavioral_score * behavioral_confidence) / (textual_confidence + behavioral_confidence)
    
#     # Calculate trust score (inverse of fake probability)
#     trust_score = int((1 - combined_score) * 100)
    
#     # Determine final prediction and confidence level
#     if combined_score >= 0.7:
#         final_prediction = 'fake'
#         confidence_level = 'high'
#         overall_reason = "Strong evidence from both textual and behavioral analysis"
#     elif combined_score >= 0.6:
#         final_prediction = 'fake'
#         confidence_level = 'medium'
#         overall_reason = "Moderate evidence indicating fake activity"
#     elif combined_score >= 0.4:
#         final_prediction = 'suspicious'
#         confidence_level = 'medium'
#         overall_reason = "Mixed signals from textual and behavioral analysis"
#     elif combined_score >= 0.3:
#         final_prediction = 'genuine'
#         confidence_level = 'medium'
#         overall_reason = "Likely genuine with some minor concerns"
#     else:
#         final_prediction = 'genuine'
#         confidence_level = 'high'
#         overall_reason = "Strong evidence indicating genuine review"
    
#     # Generate comprehensive flags and explanations
#     comprehensive_flags = generate_comprehensive_flags(textual_result, behavioral_result)
#     combined_explanation = generate_combined_explanation(textual_result, behavioral_result, final_prediction, combined_score)
    
#     return {
#         'final_prediction': final_prediction,
#         'combined_score': round(combined_score, 3),
#         'trust_score': trust_score,
#         'confidence': confidence_level,
#         'overall_reason': overall_reason,
#         'comprehensive_flags': comprehensive_flags,
#         'combined_explanation': combined_explanation,
#         'textual_analysis': {
#             'prediction': textual_pred,
#             'probability': round(textual_prob, 3),
#             'confidence': textual_result['confidence'],
#             'key_factors': textual_result['explanation']['key_factors'],
#             'warning_flags': textual_result['explanation']['warning_flags'],
#             'detailed_reasons': textual_result['explanation'].get('detailed_reasons', [])
#         },
#         'behavioral_analysis': {
#             'prediction': behavioral_pred,
#             'probability': round(behavioral_prob, 3),
#             'confidence': behavioral_result['confidence'],
#             'behavioral_flags': behavioral_result.get('behavioral_flags', []),
#             'user_metrics': behavioral_result.get('user_metrics', {}),
#             'calculated_metrics': behavioral_result.get('calculated_metrics', {})
#         }
#     }

def calculate_confidence_weight(probability):
    """Calculate confidence weight based on probability distance from 0.5"""
    distance_from_decision = abs(probability - 0.5)
    
    if distance_from_decision > 0.4:
        return 2.0  # High confidence
    elif distance_from_decision > 0.2:
        return 1.5  # Medium confidence
    else:
        return 1.0  # Low confidence

def generate_comprehensive_flags(textual_result, behavioral_result):
    """Generate comprehensive flags combining both analyses"""
    flags = []
    
    # Textual flags
    for flag in textual_result['explanation']['warning_flags']:
        flags.append({
            'type': f'textual_{flag["type"]}',
            'severity': 'medium',
            'source': 'textual_analysis',
            'description': flag['description'],
            'impact': 'Affects review content credibility'
        })
    
    # Behavioral flags
    for flag in behavioral_result.get('behavioral_flags', []):
        flags.append({
            'type': f'behavioral_{flag["type"]}',
            'severity': flag.get('severity', 'medium'),
            'source': 'behavioral_analysis',
            'description': flag.get('description', 'Behavioral pattern detected'),
            'impact': 'Affects user credibility patterns'
        })
    
    # Combined risk assessment
    textual_risk = 1 if textual_result['prediction'] == 'fake' else 0
    behavioral_risk = 1 if behavioral_result['prediction'] == 'fake' else 0
    
    if textual_risk == 1 and behavioral_risk == 1:
        flags.append({
            'type': 'high_risk_combined',
            'severity': 'high',
            'source': 'combined_analysis',
            'description': 'Both textual and behavioral analysis indicate fake activity',
            'impact': 'Very high probability of fake review'
        })
    elif textual_risk == 1 or behavioral_risk == 1:
        flags.append({
            'type': 'medium_risk_single',
            'severity': 'medium',
            'source': 'combined_analysis',
            'description': 'One analysis method indicates potential fake activity',
            'impact': 'Moderate probability of fake review'
        })
    
    return flags

def generate_combined_explanation(textual_result, behavioral_result, final_prediction, combined_score):
    """Generate comprehensive explanation combining both analyses"""
    explanation = {
        'summary': '',
        'textual_evidence': [],
        'behavioral_evidence': [],
        'recommendations': []
    }
    
    # Calculate genuine confidence for summary
    genuine_confidence = (1 - combined_score) * 100
    
    # Generate summary based on combined results
    if final_prediction == 'fake':
        explanation['summary'] = f"This review shows strong signs of being fake with {combined_score*100:.1f}% confidence. Multiple detection methods indicate artificial activity."
    elif final_prediction == 'suspicious':
        explanation['summary'] = f"This review shows suspicious characteristics with {combined_score*100:.1f}% fake probability. Some aspects appear genuine while others raise concerns."
    else:
        explanation['summary'] = f"This review appears genuine with {genuine_confidence:.1f}% confidence. Analysis indicates authentic user behavior and content."
    
    # Textual evidence
    textual_pred = textual_result['prediction']
    if textual_pred == 'fake':
        explanation['textual_evidence'].append({
            'type': 'content_analysis',
            'description': 'Review content analysis indicates artificial writing patterns',
            'confidence': textual_result['confidence'],
            'key_factors': list(textual_result['explanation']['key_factors'].keys())[:3]
        })
    else:
        explanation['textual_evidence'].append({
            'type': 'content_analysis',
            'description': 'Review content appears natural and authentic',
            'confidence': textual_result['confidence'],
            'key_factors': ['Natural language patterns', 'Appropriate sentiment']
        })
    
    # Behavioral evidence - use CORRECTED interpretation
    behavioral_flags = behavioral_result.get('behavioral_flags', [])
    has_suspicious_behavior = any(flag.get('severity') in ['high', 'medium'] for flag in behavioral_flags)
    
    if has_suspicious_behavior or behavioral_result.get('prediction') == 'fake':
        explanation['behavioral_evidence'].append({
            'type': 'user_behavior',
            'description': 'User behavior patterns indicate suspicious activity',
            'confidence': behavioral_result.get('confidence', 'medium'),
            'key_metrics': ['Review frequency', 'Account patterns', 'Rating distribution']
        })
    else:
        explanation['behavioral_evidence'].append({
            'type': 'user_behavior',
            'description': 'User behavior patterns appear normal and consistent',
            'confidence': behavioral_result.get('confidence', 'medium'),
            'key_metrics': ['Normal review frequency', 'Consistent activity patterns']
        })
    
    # Recommendations
    if final_prediction == 'fake':
        explanation['recommendations'] = [
            'Consider removing this review due to high fake probability',
            'Monitor this user for similar patterns',
            'Verify product authenticity if multiple fake reviews detected'
        ]
    elif final_prediction == 'suspicious':
        explanation['recommendations'] = [
            'Review requires manual verification',
            'Monitor user for consistent patterns',
            'Compare with other reviews from same user'
        ]
    else:
        explanation['recommendations'] = [
            'Review appears genuine - no action needed',
            'Continue normal monitoring patterns',
            'Consider featuring if high-quality content'
        ]
    
    return explanation


def store_comprehensive_analysis(text, rating, combined_result, user_id):
    """Store comprehensive analysis results in database"""
    try:
        cur = mysql.connection.cursor()
        
        # Insert comprehensive analysis
        cur.execute("""
            INSERT INTO comprehensive_analys 
            (review_text, rating, user_id, final_prediction, combined_score, trust_score,
             textual_prediction, textual_probability, behavioral_prediction, 
             behavioral_probability, comprehensive_flags, combined_explanation, created_at) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            text[:1000],
            rating if rating is not None else None,
            user_id,
            combined_result['final_prediction'],
            combined_result['combined_score'],
            combined_result['trust_score'],
            combined_result['textual_analysis']['prediction'],
            combined_result['textual_analysis']['probability'],
            combined_result['behavioral_analysis']['prediction'],
            combined_result['behavioral_analysis']['probability'],
            json.dumps(combined_result['comprehensive_flags']),
            json.dumps(combined_result['combined_explanation']),
            datetime.now()
        ))
        
        analysis_id = cur.lastrowid
        
        mysql.connection.commit()
        cur.close()
        logger.info("Successfully stored comprehensive analysis, analysis_id: %s", analysis_id)
        return True
        
    except Exception as e:
        logger.error(f"Database error in comprehensive analysis: {str(e)}")
        mysql.connection.rollback()
        return False
    
@app.route('/behavioral_model_info', methods=['GET'])
def get_behavioral_model_info():
    """Get information about the loaded behavioral model"""
    try:
        if behavioral_predictor is None:
            return jsonify({
                "error": "Behavioral model not loaded"
            }), 503
        
        model_info = {
            "model_loaded": True,
            "feature_count": len(behavioral_predictor.feature_names),
            "features": behavioral_predictor.feature_names,
            "model_type": behavioral_predictor.metadata.get('model_type', 'unknown'),
            "training_date": behavioral_predictor.metadata.get('training_date', 'unknown')
        }
        
        return jsonify({
            "success": True,
            "model_info": model_info
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            "error": "Failed to get model info",
            "message": str(e)
        }), 500
        
# Helper analysis functions
def analyze_by_readability(reviews):
    """Analyze fake reviews by readability score"""
    low_readability = sum(1 for r in reviews if r['readability_score'] < 40)
    medium_readability = sum(1 for r in reviews if 40 <= r['readability_score'] <= 70)
    high_readability = sum(1 for r in reviews if r['readability_score'] > 70)
    
    return {
        "low_readability": {
            "count": low_readability,
            "percentage": round(low_readability / len(reviews) * 100, 1),
            "description": "Complex, hard-to-read reviews (often AI-generated or overly formal)"
        },
        "medium_readability": {
            "count": medium_readability,
            "percentage": round(medium_readability / len(reviews) * 100, 1),
            "description": "Moderately readable reviews"
        },
        "high_readability": {
            "count": high_readability,
            "percentage": round(high_readability / len(reviews) * 100, 1),
            "description": "Very simple, easy-to-read reviews (often template-based or bot-generated)"
        }
    }

def analyze_by_sentiment(reviews):
    """Analyze fake reviews by sentiment"""
    # very_negative = sum(1 for r in reviews if r['sentiment_score'] < -0.5)
    negative = sum(1 for r in reviews if -0.5 <= r['sentiment_score'] < -0.3)
    neutral = sum(1 for r in reviews if -0.1 <= r['sentiment_score'] <= 0.3)
    # positive = sum(1 for r in reviews if 0.1 < r['sentiment_score'] <= 0.5)
    positive = sum(1 for r in reviews if r['sentiment_score'] > 0.3)
    
    return {
        # "very_negative": {
        #     "count": very_negative,
        #     "percentage": round(very_negative / len(reviews) * 100, 1),
        #     "description": "Extremely negative fake reviews"
        # },
        "negative": {
            "count": negative,
            "percentage": round(negative / len(reviews) * 100, 1),
            "description": "Negative fake reviews"
        },
        "neutral": {
            "count": neutral,
            "percentage": round(neutral / len(reviews) * 100, 1),
            "description": "Neutral fake reviews"
        },
        "positive": {
            "count": positive,
            "percentage": round(positive / len(reviews) * 100, 1),
            "description": "Positive fake reviews"
        }
        # ,
        # "very_positive": {
        #     "count": very_positive,
        #     "percentage": round(very_positive / len(reviews) * 100, 1),
        #     "description": "Extremely positive fake reviews (common for promotional fake reviews)"
        # }
    }

def analyze_by_length(reviews):
    """Analyze fake reviews by word count"""
    very_short = sum(1 for r in reviews if r['word_count'] < 10)
    short = sum(1 for r in reviews if 10 <= r['word_count'] < 30)
    medium = sum(1 for r in reviews if 30 <= r['word_count'] < 60)
    long = sum(1 for r in reviews if 60 <= r['word_count'] < 100)
    very_long = sum(1 for r in reviews if r['word_count'] >= 100)
    
    return {
        "very_short": {
            "count": very_short,
            "percentage": round(very_short / len(reviews) * 100, 1),
            "description": "Very short reviews (<10 words) - often bot-generated"
        },
        "short": {
            "count": short,
            "percentage": round(short / len(reviews) * 100, 1),
            "description": "Short reviews (10-30 words)"
        },
        "medium": {
            "count": medium,
            "percentage": round(medium / len(reviews) * 100, 1),
            "description": "Medium-length reviews (30-60 words)"
        },
        "long": {
            "count": long,
            "percentage": round(long / len(reviews) * 100, 1),
            "description": "Long reviews (60-100 words)"
        },
        "very_long": {
            "count": very_long,
            "percentage": round(very_long / len(reviews) * 100, 1),
            "description": "Very long reviews (100+ words) - often AI-generated"
        }
    }

def analyze_by_rating(reviews):
    """Analyze fake reviews by rating"""
    rating_counts = {}
    for i in range(1, 6):
        count = sum(1 for r in reviews if r['rating'] == i)
        rating_counts[str(i)] = {
            "count": count,
            "percentage": round(count / len(reviews) * 100, 1) if reviews else 0,
            "description": f"{i}-star fake reviews"
        }
    return rating_counts

def analyze_warning_flags(reviews):
    """Analyze common warning flags in fake reviews"""
    flag_counts = {}
    for review in reviews:
        if review['warning_flags']:
            flags = review['warning_flags'].split(',')
            for flag in flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
    
    # Sort by frequency
    sorted_flags = sorted(flag_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        flag: {
            "count": count,
            "percentage": round(count / len(reviews) * 100, 1),
            "description": get_flag_description(flag)
        }
        for flag, count in sorted_flags
    }

def get_flag_description(flag_type):
    """Get human-readable description for each flag type"""
    descriptions = {
        "sentiment_rating_mismatch": "Sentiment doesn't match the star rating",
        "overly_simple_language": "Too simple language for the content",
        "overly_complex_language": "Unnecessarily complex language",
        "excessive_first_person_usage": "Overuse of first-person pronouns",
        "excessive_superlatives": "Too many superlative words",
        "robotic_pattern": "Pattern matches known robotic writing",
        "template_like": "Appears to use a common template"
    }
    return descriptions.get(flag_type, flag_type)

def analyze_time_trends(reviews):
    """Analyze when fake reviews are being posted"""
    # Group by hour of day
    hourly_counts = {}
    for review in reviews:
        if review['created_at']:
            hour = review['created_at'].hour
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
    
    return {
        "peak_hours": sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)[:5],
        "total_by_hour": hourly_counts
    }

def get_most_common_types(reviews):
    """Identify the most common types of fake reviews"""
    common_types = []
    
    # Type 1: Overly positive fake reviews
    overly_positive = sum(1 for r in reviews if r['sentiment_score'] > 0.7 and r['rating'] >= 4)
    if overly_positive > 0:
        common_types.append({
            "type": "overly_positive_promotional",
            "count": overly_positive,
            "percentage": round(overly_positive / len(reviews) * 100, 1),
            "characteristics": ["high sentiment", "high rating", "excessive positivity"],
            "purpose": "Promote products/services"
        })
    
    # Type 2: Overly negative fake reviews
    overly_negative = sum(1 for r in reviews if r['sentiment_score'] < -0.7 and r['rating'] <= 2)
    if overly_negative > 0:
        common_types.append({
            "type": "overly_negative_attack",
            "count": overly_negative,
            "percentage": round(overly_negative / len(reviews) * 100, 1),
            "characteristics": ["low sentiment", "low rating", "excessive negativity"],
            "purpose": "Damage competitor reputation"
        })
    
    # Type 3: Robotic/simple reviews
    robotics = sum(1 for r in reviews if 
                   r.get('robotic_flag', 0) == 1 or  # Use .get() with default 0
                   (r['readability_score'] > 80 and r['word_count'] < 15))
    if robotics > 0:
        common_types.append({
            "type": "robotic_template",
            "count": robotics,
            "percentage": round(robotics / len(reviews) * 100, 1),
            "characteristics": ["high readability", "low word count", "simple language"],
            "purpose": "Mass-generated fake reviews"
        })
    
    # Type 4: Mismatched reviews
    mismatched = sum(1 for r in reviews if r.get('mismatch_flag', 0) == 1)
    if mismatched > 0:
        common_types.append({
            "type": "sentiment_rating_mismatch",
            "count": mismatched,
            "percentage": round(mismatched / len(reviews) * 100, 1),
            "characteristics": ["rating doesn't match text sentiment", "contradictory language"],
            "purpose": "Attempt to appear genuine while manipulating ratings"
        })
    
    # Sort by most common
    common_types.sort(key=lambda x: x['count'], reverse=True)
    return common_types

@app.route('/analytics/report', methods=['GET'])
def generate_analytics_report():
    """Generate comprehensive analytics report for fake/genuine reviews."""
    try:
        report_type = request.args.get('type', 'both')  # 'fake', 'genuine', or 'both'
        time_period = request.args.get('period', '7d')  # '7d', '30d', 'all'
        
        cur = mysql.connection.cursor()
        
        # Build time filter
        time_filters = {
            '7d': "created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)",
            '30d': "created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)",
            'all': "1=1"
        }
        time_filter = time_filters.get(time_period, "1=1")
        
        # Build prediction filter
        if report_type == 'fake':
            pred_filter = "prediction = 'fake'"
        elif report_type == 'genuine':
            pred_filter = "prediction = 'genuine'"
        else:
            pred_filter = "1=1"
        
        # Get basic statistics
        cur.execute(f"""
            SELECT 
                prediction,
                COUNT(*) as total_reviews,
                ROUND(AVG(probability), 3) as avg_confidence,
                ROUND(AVG(word_count), 1) as avg_word_count,
                ROUND(AVG(sentiment_score), 3) as avg_sentiment,
                ROUND(AVG(readability_score), 1) as avg_readability,
                SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as rating_1,
                SUM(CASE WHEN rating = 2 THEN 1 ELSE 0 END) as rating_2,
                SUM(CASE WHEN rating = 3 THEN 1 ELSE 0 END) as rating_3,
                SUM(CASE WHEN rating = 4 THEN 1 ELSE 0 END) as rating_4,
                SUM(CASE WHEN rating = 5 THEN 1 ELSE 0 END) as rating_5
            FROM reviews1 
            WHERE {time_filter} AND {pred_filter}
            GROUP BY prediction
        """)
        stats = cur.fetchall()
        
        # Get daily trend
        cur.execute(f"""
            SELECT 
                DATE(created_at) as date,
                prediction,
                COUNT(*) as count
            FROM reviews1
            WHERE {time_filter} AND {pred_filter}
            GROUP BY DATE(created_at), prediction
            ORDER BY date
        """)
        daily_trend = cur.fetchall()
        
        # Get warning flags distribution
        cur.execute(f"""
            SELECT 
                wf.flag_type,
                COUNT(*) as count,
                r.prediction
            FROM warning_flags1 wf
            JOIN reviews1 r ON wf.review_id = r.review_id
            WHERE {time_filter} AND {pred_filter}
            GROUP BY wf.flag_type, r.prediction
            ORDER BY count DESC
        """)
        warning_flags = cur.fetchall()
        
        # Get sentiment distribution
        cur.execute(f"""
            SELECT 
                prediction,
                CASE 
                    WHEN sentiment_score > 0.3 THEN 'Positive'
                    WHEN sentiment_score < -0.3 THEN 'Negative'
                    ELSE 'Neutral'
                END as sentiment_category,
                COUNT(*) as count,
                ROUND(AVG(probability), 3) as avg_confidence
            FROM reviews1
            WHERE {time_filter} AND {pred_filter}
            GROUP BY prediction, sentiment_category
            ORDER BY prediction, count DESC
        """)
        sentiment_stats = cur.fetchall()
        
        # Get readability analysis
        cur.execute(f"""
            SELECT 
                prediction,
                CASE 
                    WHEN readability_score < 40 THEN 'Low'
                    WHEN readability_score BETWEEN 40 AND 70 THEN 'Medium'
                    ELSE 'High'
                END as readability_level,
                COUNT(*) as count,
                ROUND(AVG(word_count), 1) as avg_word_count
            FROM reviews1
            WHERE {time_filter} AND {pred_filter}
            GROUP BY prediction, readability_level
            ORDER BY prediction, count DESC
        """)
        readability_stats = cur.fetchall()
        
        cur.close()
        
        # Compile comprehensive report (without charts)
        report = {
            "metadata": {
                "report_type": report_type,
                "time_period": time_period,
                "generated_at": datetime.now().isoformat(),
                "total_reviews_analyzed": sum([s['total_reviews'] for s in stats])
            },
            "summary_statistics": compile_summary_statistics(stats),
            "temporal_analysis": compile_temporal_analysis(daily_trend),
            "warning_analysis": compile_warning_analysis(warning_flags),
            "sentiment_analysis": compile_sentiment_analysis(sentiment_stats),
            "readability_analysis": compile_readability_analysis(readability_stats)
        }
        
        return jsonify({
            "success": True,
            "report": report
        })
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({
            "error": "Report generation failed",
            "message": str(e)
        }), 500
        
def compile_summary_statistics(stats):
    """Compile summary statistics from database results."""
    summary = {}
    for stat in stats:
        pred = stat['prediction']
        summary[pred] = {
            "total_reviews": stat['total_reviews'],
            "average_confidence": float(stat['avg_confidence']),
            "average_word_count": float(stat['avg_word_count']),
            "average_sentiment": float(stat['avg_sentiment']),
            "average_readability": float(stat['avg_readability']),
            "rating_distribution": {
                "1_star": stat['rating_1'],
                "2_star": stat['rating_2'],
                "3_star": stat['rating_3'],
                "4_star": stat['rating_4'],
                "5_star": stat['rating_5']
            }
        }
    return summary

def compile_temporal_analysis(daily_trend):
    """Compile temporal analysis from daily trend data."""
    analysis = {}
    for trend in daily_trend:
        date_str = trend['date'].strftime("%Y-%m-%d")
        pred = trend['prediction']
        
        if date_str not in analysis:
            analysis[date_str] = {}
        
        analysis[date_str][pred] = trend['count']
    
    return analysis

def compile_warning_analysis(warning_flags):
    """Compile warning flag analysis."""
    analysis = {}
    for flag in warning_flags:
        pred = flag['prediction']
        flag_type = flag['flag_type']
        
        if pred not in analysis:
            analysis[pred] = {}
        
        analysis[pred][flag_type] = flag['count']
    
    return analysis

def compile_sentiment_analysis(sentiment_stats):
    """Compile sentiment analysis."""
    analysis = {}
    for stat in sentiment_stats:
        pred = stat['prediction']
        sentiment = stat['sentiment_category']
        
        if pred not in analysis:
            analysis[pred] = {}
        
        analysis[pred][sentiment] = {
            "count": stat['count'],
            "average_confidence": float(stat['avg_confidence'])
        }
    
    return analysis

def compile_readability_analysis(readability_stats):
    """Compile readability analysis."""
    analysis = {}
    for stat in readability_stats:
        pred = stat['prediction']
        level = stat['readability_level']
        
        if pred not in analysis:
            analysis[pred] = {}
        
        analysis[pred][level] = {
            "count": stat['count'],
            "average_word_count": float(stat['avg_word_count'])
        }
    
    return analysis

def generate_report_charts(stats, daily_trend, warning_flags, sentiment_stats, readability_stats):
    """Generate charts for the report and return as base64 encoded images."""
    charts = {}
    
    try:
        # Chart 1: Prediction Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        predictions = [s['prediction'] for s in stats]
        counts = [s['total_reviews'] for s in stats]
        
        ax.bar(predictions, counts, color=['green', 'red'])
        ax.set_title('Review Prediction Distribution')
        ax.set_ylabel('Number of Reviews')
        
        charts['prediction_distribution'] = fig_to_base64(fig)
        plt.close(fig)
        
        # Chart 2: Daily Trend
        fig, ax = plt.subplots(figsize=(12, 6))
        dates = sorted(set([t['date'] for t in daily_trend]))
        
        fake_counts = []
        genuine_counts = []
        for date in dates:
            fake = next((t['count'] for t in daily_trend if t['date'] == date and t['prediction'] == 'fake'), 0)
            genuine = next((t['count'] for t in daily_trend if t['date'] == date and t['prediction'] == 'genuine'), 0)
            fake_counts.append(fake)
            genuine_counts.append(genuine)
        
        date_labels = [d.strftime("%m-%d") for d in dates]
        ax.plot(date_labels, fake_counts, label='Fake', color='red')
        ax.plot(date_labels, genuine_counts, label='Genuine', color='green')
        ax.set_title('Daily Review Trend')
        ax.set_ylabel('Number of Reviews')
        ax.legend()
        plt.xticks(rotation=45)
        
        charts['daily_trend'] = fig_to_base64(fig)
        plt.close(fig)
        
        # Chart 3: Warning Flags
        fig, ax = plt.subplots(figsize=(12, 6))
        flag_types = sorted(set([f['flag_type'] for f in warning_flags]))
        
        fake_flags = []
        genuine_flags = []
        for flag in flag_types:
            fake = next((f['count'] for f in warning_flags if f['flag_type'] == flag and f['prediction'] == 'fake'), 0)
            genuine = next((f['count'] for f in warning_flags if f['flag_type'] == flag and f['prediction'] == 'genuine'), 0)
            fake_flags.append(fake)
            genuine_flags.append(genuine)
        
        x = range(len(flag_types))
        ax.bar([i - 0.2 for i in x], fake_flags, width=0.4, label='Fake', color='red')
        ax.bar([i + 0.2 for i in x], genuine_flags, width=0.4, label='Genuine', color='green')
        ax.set_title('Warning Flags Distribution')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels(flag_types, rotation=45, ha='right')
        ax.legend()
        
        charts['warning_flags'] = fig_to_base64(fig)
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Error generating charts: {str(e)}")
        charts['error'] = f"Chart generation failed: {str(e)}"
    
    return charts

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 encoded image."""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route('/analytics/export', methods=['GET'])
def export_analytics_report():
    """Export analytics report as CSV or JSON."""
    try:
        export_format = request.args.get('format', 'json')  # 'json' or 'csv'
        report_type = request.args.get('type', 'both')
        time_period = request.args.get('period', '7d')
        
        # Generate the report
        report_response = generate_analytics_report()
        report_data = report_response.get_json()
        
        if not report_data or 'report' not in report_data:
            return jsonify({"error": "Failed to generate report"}), 500
        
        report = report_data['report']
        
        if export_format == 'csv':
            # Create CSV data
            csv_data = []
            
            # Summary statistics
            csv_data.append(["SUMMARY STATISTICS"])
            csv_data.append(["Prediction", "Total Reviews", "Avg Confidence", "Avg Word Count", "Avg Sentiment", "Avg Readability"])
            for pred, stats in report['summary_statistics'].items():
                csv_data.append([
                    pred,
                    stats['total_reviews'],
                    stats['average_confidence'],
                    stats['average_word_count'],
                    stats['average_sentiment'],
                    stats['average_readability']
                ])
            
            csv_data.append([])
            csv_data.append(["TEMPORAL ANALYSIS"])
            csv_data.append(["Date", "Fake Reviews", "Genuine Reviews"])
            for date, counts in report['temporal_analysis'].items():
                csv_data.append([
                    date,
                    counts.get('fake', 0),
                    counts.get('genuine', 0)
                ])
            
            # Convert to CSV string
            csv_output = "\n".join([",".join(map(str, row)) for row in csv_data])
            
            return csv_output, 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': f'attachment; filename=analytics_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            }
        
        else:  # JSON format
            return jsonify({
                "success": True,
                "report": report
            })
            
    except Exception as e:
        logger.error(f"Error exporting report: {str(e)}")
        return jsonify({
            "error": "Export failed",
            "message": str(e)
        }), 500
@app.route('/analytics/dashboard', methods=['GET'])
def get_dashboard_analytics():
    """
    Returns comprehensive analytics for dashboard:
    - Total analyses count
    - Prediction distribution
    - Daily trends
    - Accuracy metrics
    - Recent activity
    """
    try:
        cur = mysql.connection.cursor()
        
        # Overall statistics
        cur.execute("""
            SELECT 
                COUNT(*) as total_analyses,
                SUM(CASE WHEN final_prediction = 'fake' THEN 1 ELSE 0 END) as fake_count,
                SUM(CASE WHEN final_prediction = 'genuine' THEN 1 ELSE 0 END) as genuine_count,
                SUM(CASE WHEN final_prediction = 'suspicious' THEN 1 ELSE 0 END) as suspicious_count,
                ROUND(AVG(combined_score), 3) as avg_confidence,
                ROUND(AVG(trust_score), 1) as avg_trust_score
            FROM comprehensive_analys
        """)
        overall_stats = cur.fetchone()
        
        # Daily trend (last 30 days)
        cur.execute("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as total_analyses,
                SUM(CASE WHEN final_prediction = 'fake' THEN 1 ELSE 0 END) as fake_count,
                SUM(CASE WHEN final_prediction = 'genuine' THEN 1 ELSE 0 END) as genuine_count,
                ROUND(AVG(combined_score), 3) as avg_confidence
            FROM comprehensive_analys 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            LIMIT 30
        """)
        daily_trends = cur.fetchall()
        
        # Behavioral flags distribution
        cur.execute("""
            SELECT 
                JSON_EXTRACT(comprehensive_flags, '$[*].type') as flag_types,
                COUNT(*) as frequency
            FROM comprehensive_analys 
            WHERE comprehensive_flags IS NOT NULL
            GROUP BY flag_types
            ORDER BY frequency DESC
            LIMIT 10
        """)
        common_flags = cur.fetchall()
        
        # Recent analyses with details
        cur.execute("""
            SELECT 
                ca.analysis_id,
                ca.review_text,
                ca.final_prediction,
                ca.combined_score,
                ca.trust_score,
                ca.created_at,
                ca.user_id,
                LENGTH(ca.review_text) as text_length
            FROM comprehensive_analys ca
            ORDER BY ca.created_at DESC
            LIMIT 10
        """)
        recent_analyses = cur.fetchall()
        
        cur.close()
        
        return jsonify({
            "success": True,
            "dashboard_analytics": {
                "overall_statistics": {
                    "total_analyses": overall_stats['total_analyses'],
                    "fake_count": overall_stats['fake_count'],
                    "genuine_count": overall_stats['genuine_count'],
                    "suspicious_count": overall_stats['suspicious_count'],
                    "fake_percentage": round((overall_stats['fake_count'] / overall_stats['total_analyses']) * 100, 1) if overall_stats['total_analyses'] > 0 else 0,
                    "avg_confidence": float(overall_stats['avg_confidence']),
                    "avg_trust_score": float(overall_stats['avg_trust_score'])
                },
                "daily_trends": [
                    {
                        "date": trend['date'].strftime("%Y-%m-%d"),
                        "total_analyses": trend['total_analyses'],
                        "fake_count": trend['fake_count'],
                        "genuine_count": trend['genuine_count'],
                        "fake_ratio": round(trend['fake_count'] / trend['total_analyses'], 3) if trend['total_analyses'] > 0 else 0,
                        "avg_confidence": float(trend['avg_confidence'])
                    }
                    for trend in daily_trends
                ],
                "common_behavioral_flags": [
                    {
                        "flag_type": flag['flag_types'],
                        "frequency": flag['frequency']
                    }
                    for flag in common_flags
                ],
                "recent_analyses": [
                    {
                        "analysis_id": analysis['analysis_id'],
                        "review_preview": analysis['review_text'][:100] + "..." if analysis['review_text'] and len(analysis['review_text']) > 100 else analysis['review_text'],
                        "prediction": analysis['final_prediction'],
                        "confidence": float(analysis['combined_score']),
                        "trust_score": analysis['trust_score'],
                        "text_length": analysis['text_length'],
                        "created_at": analysis['created_at'].strftime("%Y-%m-%d %H:%M:%S"),
                        "user_id": analysis['user_id']
                    }
                    for analysis in recent_analyses
                ]
            }
        })
        
    except Exception as e:
        logger.error(f"Dashboard analytics error: {str(e)}")
        return jsonify({"error": "Failed to fetch dashboard analytics"}), 500
@app.route('/analytics/historical', methods=['GET'])
def get_historical_analytics():
    """
    Advanced historical analysis with filters:
    - Date range
    - Prediction type
    - Confidence thresholds
    - User segments
    """
    try:
        # Get query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        prediction_type = request.args.get('prediction', 'all')  # all, fake, genuine, suspicious
        min_confidence = float(request.args.get('min_confidence', 0.0))
        min_trust_score = int(request.args.get('min_trust_score', 0))
        
        cur = mysql.connection.cursor()
        
        # Build dynamic WHERE clause
        where_conditions = ["1=1"]
        params = []
        
        if start_date:
            where_conditions.append("created_at >= %s")
            params.append(start_date)
        if end_date:
            where_conditions.append("created_at <= %s")
            params.append(end_date)
        if prediction_type != 'all':
            where_conditions.append("final_prediction = %s")
            params.append(prediction_type)
        if min_confidence > 0:
            where_conditions.append("combined_score >= %s")
            params.append(min_confidence)
        if min_trust_score > 0:
            where_conditions.append("trust_score >= %s")
            params.append(min_trust_score)
        
        where_clause = " AND ".join(where_conditions)
        
        # Historical trend by week
        cur.execute(f"""
            SELECT 
                YEAR(created_at) as year,
                WEEK(created_at) as week,
                COUNT(*) as total_analyses,
                SUM(CASE WHEN final_prediction = 'fake' THEN 1 ELSE 0 END) as fake_count,
                SUM(CASE WHEN final_prediction = 'genuine' THEN 1 ELSE 0 END) as genuine_count,
                ROUND(AVG(combined_score), 3) as avg_confidence,
                ROUND(AVG(trust_score), 1) as avg_trust_score,
                MIN(created_at) as week_start
            FROM comprehensive_analys 
            WHERE {where_clause}
            GROUP BY YEAR(created_at), WEEK(created_at)
            ORDER BY year DESC, week DESC
            LIMIT 52
        """, params)
        weekly_trends = cur.fetchall()
        
        # Prediction accuracy over time (if you have validation data)
        cur.execute(f"""
            SELECT 
                DATE(created_at) as date,
                final_prediction,
                COUNT(*) as count,
                ROUND(AVG(combined_score), 3) as avg_confidence
            FROM comprehensive_analys 
            WHERE {where_clause}
            GROUP BY DATE(created_at), final_prediction
            ORDER BY date DESC
            LIMIT 100
        """, params)
        prediction_evolution = cur.fetchall()
        
        # User behavior patterns
        cur.execute(f"""
            SELECT 
                user_id,
                COUNT(*) as analysis_count,
                SUM(CASE WHEN final_prediction = 'fake' THEN 1 ELSE 0 END) as fake_count,
                ROUND(AVG(combined_score), 3) as avg_confidence,
                ROUND(AVG(trust_score), 1) as avg_trust_score,
                MAX(created_at) as last_analysis
            FROM comprehensive_analys 
            WHERE {where_clause} AND user_id IS NOT NULL
            GROUP BY user_id
            HAVING analysis_count >= 1
            ORDER BY analysis_count DESC
            LIMIT 50
        """, params)
        user_patterns = cur.fetchall()
        
        # Confidence distribution
        cur.execute(f"""
            SELECT 
                CASE 
                    WHEN combined_score < 0.3 THEN 'Low (0-0.3)'
                    WHEN combined_score < 0.6 THEN 'Medium (0.3-0.6)'
                    WHEN combined_score < 0.8 THEN 'High (0.6-0.8)'
                    ELSE 'Very High (0.8-1.0)'
                END as confidence_bucket,
                COUNT(*) as count,
                ROUND(AVG(trust_score), 1) as avg_trust_score
            FROM comprehensive_analys 
            WHERE {where_clause}
            GROUP BY confidence_bucket
            ORDER BY MIN(combined_score)
        """, params)
        confidence_distribution = cur.fetchall()
        
        cur.close()
        
        return jsonify({
            "success": True,
            "filters_applied": {
                "start_date": start_date,
                "end_date": end_date,
                "prediction_type": prediction_type,
                "min_confidence": min_confidence,
                "min_trust_score": min_trust_score
            },
            "historical_analytics": {
                "weekly_trends": [
                    {
                        "year": trend['year'],
                        "week": trend['week'],
                        "week_start": trend['week_start'].strftime("%Y-%m-%d"),
                        "total_analyses": trend['total_analyses'],
                        "fake_count": trend['fake_count'],
                        "genuine_count": trend['genuine_count'],
                        "fake_ratio": round(trend['fake_count'] / trend['total_analyses'], 3) if trend['total_analyses'] > 0 else 0,
                        "avg_confidence": float(trend['avg_confidence']),
                        "avg_trust_score": float(trend['avg_trust_score'])
                    }
                    for trend in weekly_trends
                ],
                "prediction_evolution": [
                    {
                        "date": evolution['date'].strftime("%Y-%m-%d"),
                        "prediction": evolution['final_prediction'],
                        "count": evolution['count'],
                        "avg_confidence": float(evolution['avg_confidence'])
                    }
                    for evolution in prediction_evolution
                ],
                "user_behavior_patterns": [
                    {
                        "user_id": pattern['user_id'],
                        "analysis_count": pattern['analysis_count'],
                        "fake_count": pattern['fake_count'],
                        "fake_ratio": round(pattern['fake_count'] / pattern['analysis_count'], 3) if pattern['analysis_count'] > 0 else 0,
                        "avg_confidence": float(pattern['avg_confidence']),
                        "avg_trust_score": float(pattern['avg_trust_score']),
                        "last_analysis": pattern['last_analysis'].strftime("%Y-%m-%d %H:%M:%S")
                    }
                    for pattern in user_patterns
                ],
                "confidence_distribution": [
                    {
                        "confidence_range": dist['confidence_bucket'],
                        "count": dist['count'],
                        "percentage": round((dist['count'] / sum(d['count'] for d in confidence_distribution)) * 100, 1) if confidence_distribution else 0,
                        "avg_trust_score": float(dist['avg_trust_score'])
                    }
                    for dist in confidence_distribution
                ]
            }
        })
        
    except Exception as e:
        logger.error(f"Historical analytics error: {str(e)}")
        return jsonify({"error": "Failed to fetch historical analytics"}), 500
                
if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)