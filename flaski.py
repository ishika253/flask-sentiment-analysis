from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the sentiment model
model = load_model('C:/Users/raiya/Desktop/LLM model/ChatGPT_files/model.keras')

# Load the dataset to fit the tokenizer (same as training)
df = pd.read_excel('C:/Users/raiya/Desktop/LLM model/ChatGPT_files/sentiment_analysis_results.xlsx')
texts = df['Query'].values
labels = df['sentiment'].values
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)

# Fit the tokenizer
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(labels)


# Define max length for padding
max_length = 20

def preprocess_input(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length)
    return padded



@app.route('/')
def home():
    return "Flask server is running!"


@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    
    if req is None:
        return jsonify({'fulfillmentText': 'Invalid request, no JSON payload received'}), 400
    
    # Adjusted to handle simplified payload
    user_input = req.get('queryText')
    if user_input is None:
        return jsonify({'fulfillmentText': 'Invalid request, queryText not found'}), 400

    processed_input = preprocess_input(user_input)
    sentiment_score = model.predict(processed_input)[0][0]

    
    if sentiment_score > 0.6:
        response = "I'm glad to hear that! 😊 How can I assist you further?"
    elif sentiment_score < 0.4:
        response = "I'm sorry to hear that. 😟 Is there anything specific you'd like to talk about?"
    else:
        response = "Thanks for sharing! What would you like to discuss?"

    return jsonify({'fulfillmentText': response})

if __name__ == '__main__':
    app.run(debug=True)
