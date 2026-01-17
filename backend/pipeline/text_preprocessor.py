"""
Text Preprocessing Module
Cleans and normalizes text using NLTK and spaCy
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Clean and normalize text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s\.]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text):
    """
    Full text preprocessing pipeline
    """
    if not text or len(text.strip()) == 0:
        return ""
    
    # Clean text
    text = clean_text(text)
    
    if not text:
        return ""
    
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    processed_sentences = []
    for sentence in sentences:
        # Tokenize words
        words = word_tokenize(sentence.lower())
        
        # Remove stopwords and lemmatize
        filtered_words = [
            lemmatizer.lemmatize(word) 
            for word in words 
            if word.isalpha() and word not in stop_words and len(word) > 2
        ]
        
        if filtered_words:  # Only keep non-empty sentences
            processed_sentences.append(' '.join(filtered_words))
    
    return ' '.join(processed_sentences)

def extract_key_phrases(text):
    """
    Extract key phrases and entities from text using spaCy
    """
    if not nlp or not text:
        return []
    
    doc = nlp(text)
    phrases = []
    
    # Extract noun chunks
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 4:  # Only phrases up to 4 words
            phrases.append(chunk.text.lower())
    
    # Extract named entities
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
            phrases.append(ent.text.lower())
    
    return list(set(phrases))  # Remove duplicates

def segment_into_sentences(text):
    """
    Segment text into sentences
    """
    if not text:
        return []
    
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]
