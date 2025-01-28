import gradio as gr
import joblib
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load the model
model = joblib.load('best_performing_model.joblib')

def detect_brand(text):
    """Detect brand mentioned in the text."""
    brand_keywords = {
        'Apple': ['iphone', 'ios', 'macbook', 'mac', 'apple', 'ipad'],
        'Google': ['google', 'android', 'pixel', 'chrome', 'nexus']
    }
    
    text_lower = text.lower()
    detected_brand = next((brand for brand, keywords in brand_keywords.items() 
                            if any(keyword in text_lower for keyword in keywords)), 'Not Detected')
    
    return detected_brand

def preprocess_text(text):
    """Preprocess input text."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\brt\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_lemmatize(text):
    """Tokenize and lemmatize preprocessed text."""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def predict_sentiment_interface(text):
    """Main prediction function for the interface."""
    if not text:
        return "Please enter some text to analyze."
    
    try:
        # Preprocess and predict
        processed_text = preprocess_text(text)
        processed_text = tokenize_and_lemmatize(processed_text)
        prediction = model.predict([processed_text])[0]
        probabilities = model.predict_proba([processed_text])[0]
        
        # Get confidence score for the predicted sentiment
        sentiment_labels = model.classes_
        prediction_index = list(sentiment_labels).index(prediction)
        confidence_score = probabilities[prediction_index]
        
        # Detect brand
        brand = detect_brand(text)
        
        # Format output
        result = f"""🎯 Sentiment: {prediction}

📱 Brand Detected: {brand}

📊 Confidence Score: {confidence_score:.1%}"""
            
        return result
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Define examples
examples = [
    ["I absolutely love my new iPhone 14 Pro! The camera quality is mind-blowing! #Apple"],
    ["The new MacBook is okay, but it's a very expensive for what you get."],
    ["Chrome keeps crashing on my laptop, very frustrating experience."],
    ["Just got the new Pixel 7, it's okay I guess. Some features are nothing special, others need work."],
    ["This Android 13 update is terrible, my battery life is worse than ever. @Google fix this!"]
]

# Create Gradio app
iface = gr.Interface(
    fn=predict_sentiment_interface,
    inputs=gr.Textbox(
        lines=3, 
        placeholder="Enter your tweet here...",
        label="Product Review"
    ),
    outputs=gr.Textbox(
        label="Analysis Result", 
        lines=6
    ),
    title=" 💭 Tech Product Sentiment Analyzer",
    description="Analyze sentiment and detect brands in tweets about Apple and Google products 📱",
    theme=gr.themes.Soft(),
    examples=examples,
    examples_per_page=5,
    cache_examples=True,
)



# Launch the app
if __name__ == "__main__":
    iface.launch()