from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS to handle cross-origin requests
import random
import time  # Import the time module

# Initialize Flask app
app = Flask(__name__)

# Explicitly allow CORS for the client origin (http://10.220.2.185:3000)
CORS(app, resources={r"/*": {"origins": "http://10.220.2.185:3000"}})

# Load variations from a file
with open('variations.txt', 'r') as f:
    variations = [line.strip() for line in f.readlines()]

# Shuffle the variations and keep track of used ones
random.shuffle(variations)
used_variations = set()

# Define a function to get a new variation
def get_new_variation():
    global variations, used_variations
    
    # If all variations have been used, reshuffle and reset
    if len(used_variations) == len(variations):
        used_variations.clear()
        random.shuffle(variations)
    
    # Get a variation that hasn't been used yet
    for variation in variations:
        if variation not in used_variations:
            used_variations.add(variation)
            return variation

# Define the route for generating a response
@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        time.sleep(5)

        # Get a new variation from the list
        response = get_new_variation()

        # Return the response as JSON
        return jsonify({"response": response})

    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": str(e)})

# Start the Flask app on port 5001
if __name__ == '__main__':
    print("Starting Flask server on port 5001...")
    app.run(host='0.0.0.0', port=5001)  # Running on port 5001
