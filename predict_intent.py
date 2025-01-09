import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)

# Enable CORS for your frontend at http://10.220.2.185:3000
CORS(app, resources={r"/*": {"origins": "http://10.220.2.185:3000"}})


with tf.device('/CPU:0'):
    model = tf.keras.models.load_model('intent_recognition_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
    binarizer = joblib.load('label_binarizer.pkl')

# Preprocess and predict functions
def preprocess_input(texts):
    return tf.convert_to_tensor(texts, dtype=tf.string)

def predict_intent(texts, threshold=0.5):

    processed_texts = preprocess_input(texts)

    predictions = model(processed_texts)

    probabilities = tf.nn.softmax(predictions).numpy()

    predicted_classes = []

    for prob in probabilities:
        # Check if any class probability exceeds the threshold
        max_prob = max(prob)
        if max_prob < threshold:
            predicted_classes.append("other")
        else:
            # If above the threshold, get the corresponding intent
            predicted_class = binarizer.classes_[prob.argmax()]
            predicted_classes.append(predicted_class)

    return predicted_classes

# Handle POST requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        texts = data.get('texts')
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400

        predicted_intents = predict_intent(texts)
        return jsonify({'texts': texts, 'predicted_intents': predicted_intents})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Handle CORS preflight requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://10.220.2.185:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
