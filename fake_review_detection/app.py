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
from sklearn.feature_extraction.text import TfidfVectorizer
import logging  # Add logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Dilshara224#' 
app.config['MYSQL_DB'] = 'review_analysis'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

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
    logger.info("✅ Model, metadata, and TF-IDF vectorizer loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model/vectorizer: {str(e)}")
    raise

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
    if not isinstance(text, str) or not text.strip():
        return 0.0
    
    try:
        analysis = TextBlob(text)
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

def extract_linguistic_features(text, rating=None):
    """
    Extract comprehensive linguistic features that match the model's training features.
    Includes TF-IDF and robotic_flag/mismatch_flag.
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
    features = {
        'word_count': word_count,
        'sentence_count': len(list(doc.sents)) or 1,
        'avg_word_length': np.mean([len(t.text) for t in doc]) if doc else 0,
        'readability_score': readability,
        'sentiment_score': get_sentiment(text),
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
             adv_ratio, noun_ratio, verb_ratio, mismatch_score, created_at) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            datetime.datetime.now()
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

def predict_review(text, rating=None):
    try:
        logger.debug("Starting prediction for text: %s", text[:50] + "..." if len(text) > 50 else text)
        
        # Clean and preprocess the text
        cleaned_text = clean_text(text)
        if not cleaned_text:
            logger.warning("Cleaned text is empty for input: %s", text[:50])
        
        # Extract all linguistic features
        features = extract_linguistic_features(cleaned_text, rating)
        logger.debug("Extracted features: %s", list(features.keys()))
        
        # Create a DataFrame with all expected features
        features_df = pd.DataFrame([features])
        
        # Ensure all model-expected features are present
        missing_cols = set(feature_cols) - set(features_df.columns)
        if missing_cols:
            logger.warning("Missing features: %s, setting to 0", missing_cols)
            for col in missing_cols:
                features_df[col] = 0.0
        
        # Reorder columns
        features_df = features_df[feature_cols]
        logger.debug("Feature DataFrame shape: %s", features_df.shape)
        
        # Scale features
        try:
            features_scaled = scaler.transform(features_df)
        except Exception as e:
            logger.error(f"Error in scaler.transform: {str(e)}")
            raise
        
        # Make prediction
        try:
            prediction_proba = model.predict(features_scaled, verbose=0)[0][0]
        except Exception as e:
            logger.error(f"Error in model.predict: {str(e)}")
            raise
        
        prediction = 1 if prediction_proba >= 0.5 else 0
        
        # Explanation and warning flags
        explanation = {
            "key_factors": {
                "readability": {"value": features['readability_score'], "description": "Measures how easy the text is to read"},
                "sentiment": {"value": features['sentiment_score'], "description": "Positive/negative sentiment strength (-1 to 1)"},
                "word_count": {"value": features['word_count'], "description": "Number of words in the review"},
                "first_person_usage": {"value": features['first_person_pronouns'], "description": "Count of 'I', 'me', 'my' pronouns"},
                "superlatives": {"value": features['superlative_words'], "description": "Count of superlative words (best, worst, etc.)"}
            },
            "warning_flags": []
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
        
        result = {
            "prediction": "fake" if prediction == 1 else "genuine",
            "probability": float(prediction_proba if prediction == 1 else 1 - prediction_proba),
            "confidence": "high" if abs(prediction_proba - 0.5) > 0.3 else "medium" if abs(prediction_proba - 0.5) > 0.15 else "low",
            "explanation": explanation,
            "features": features
        }
        
        # Store in database
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

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)