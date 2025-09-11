import os
import re
import numpy as np
import spacy
from textstat import flesch_reading_ease
from textblob import TextBlob
from sentence_transformers import SentenceTransformer  # NEW: Import for BERT embeddings

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

CONTRACTIONS = {
    "don't": "do not", "can't": "cannot", "won't": "will not",
    "it's": "it is", "i'm": "i am", "they're": "they are"
}

def clean_text(text):
    """Enhanced text cleaning with contraction handling."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # Handle contractions
    for key, value in CONTRACTIONS.items():
        text = text.replace(key, value)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Lemmatization and stopword removal
    doc = nlp(text)
    cleaned_text = " ".join([
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct
    ])
    
    return cleaned_text or ""

def extract_linguistic_features(text):
    """Extract comprehensive linguistic features that match training features."""
    if not isinstance(text, str) or not text.strip():
        text = ""
    
    doc = nlp(text)
    
    try:
        readability = flesch_reading_ease(text)
    except:
        readability = 0
    
    # Calculate all features that were used during training
    features = {
        'word_count': len([t for t in doc if not t.is_punct]),
        'sentence_count': len(list(doc.sents)),
        'avg_word_length': np.mean([len(t.text) for t in doc]) if doc else 0,
        'readability_score': readability,
        'sentiment_score': TextBlob(text).sentiment.polarity,
        'exclamation_count': text.count('!'),
        'first_person_pronouns': sum(1 for t in doc if t.lemma_ in ['i', 'me', 'my']),
        'superlative_words': sum(1 for t in doc if t.tag_ in ['JJS', 'RBS']),
        'adj_ratio': sum(1 for t in doc if t.pos_ == 'ADJ') / max(1, len(doc)),
        'adv_ratio': sum(1 for t in doc if t.pos_ == 'ADV') / max(1, len(doc)),
        'noun_ratio': sum(1 for t in doc if t.pos_ == 'NOUN') / max(1, len(doc)),
        'verb_ratio': sum(1 for t in doc if t.pos_ == 'VERB') / max(1, len(doc))
    }
    
    return features

def flag_robotic_reviews(text, readability_score, word_count):
    """Flag suspicious reviews based on readability."""
    if not isinstance(text, str):
        return 0
    if readability_score > 80 and word_count < 15:
        return 1  # Too simple
    elif readability_score < 30 and word_count > 50:
        return 1  # Overly complex
    return 0

def extract_bert_embeddings(texts, batch_size=32):
    """Extract BERT embeddings in batch for efficiency. NEW: Function for BERT."""
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')  # 768-dim model
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings