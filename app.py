from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Load the saved Logistic Regression model and associated preprocessing transformers
with open('reg_classifier.pkl', 'rb') as f:
    reg_classifier = pickle.load(f)

# Load the vectorizer and selector used for preprocessing
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('selector.pkl', 'rb') as f:
    selector = pickle.load(f)

def convert_response(code):
  if code == 0:
    return "not reject"
  elif code == 1:
    return "reject email"
  else:
    return "Cant determine"

def transform_text_for_check(email, vectorizer, selector):
    # Tokenize the email
    tokens = word_tokenize(email)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]

    # Transform into TF-IDF representation
    transformed_email = vectorizer.transform([" ".join(tokens)])
    transformed_email = selector.transform(transformed_email).toarray()

    return transformed_email

# Define endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get email from request
    email = request.json['email']
    # string_email = """"""

    # Preprocess the email
    
    transformed_email = transform_text_for_check(email, vectorizer, selector)

    # Make prediction
    prediction = reg_classifier.predict(transformed_email)
    # 

    # Return prediction
    return jsonify({'status': convert_response(int(prediction[0]))})

if __name__ == '__main__':
    app.run(debug=True, port=80, host='0.0.0.0') #this is the original to be deployed on render ..
    # app.run(debug=True)#for development purposes ..
