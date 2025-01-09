from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS to handle cross-origin requests
import torch
from transformers import GPTJForCausalLM, AutoTokenizer
import re

# Initialize Flask app
app = Flask(__name__)

# Explicitly allow CORS for the client origin (http://10.220.2.185:3000)
CORS(app, resources={r"/*": {"origins": "http://10.220.2.185:3000"}})

# Load GPT-J model and tokenizer
print("Loading the GPT-J 6B model and tokenizer...")
try:
    model_name = "EleutherAI/gpt-j-6B"
    model = GPTJForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def trim_to_complete_sentence(text):
    """
    Trim the generated text to the last complete sentence.
    """
    # Regular expression to find the last complete sentence ending with a punctuation mark
    match = re.search(r'(.*?)([.!?])(\s|$)', text[::-1])  # Reverse search for last sentence-ending punctuation
    if match:
        # Reverse the match back and return the trimmed result, ensuring it includes the sentence-ending punctuation
        return text[:len(text) - match.start()][::-1]
    return text  # If no punctuation is found, return the original text

# Define the route for generating a response
@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        # Get user input from the request
        data = request.json
        user_input = data.get("message", "")

        # Tokenize the input
        input_ids = tokenizer.encode(user_input, return_tensors='pt')

        # Create an attention mask (non-zero values indicate which tokens should be attended to)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

        # Generate a response from GPT-J with the attention mask
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,  # Pass the attention mask
            max_length=40,  # Increase length to get longer, more complete responses
            pad_token_id=tokenizer.eos_token_id,  # Use eos_token_id as pad token
            do_sample=True,  # Enable sampling for more creative responses
            temperature=0.4,  # Adjust temperature for better coherence (lower temperature = more focused)
            top_k=50,  # Top-k sampling
            top_p=0.9,  # Nucleus sampling (increase to allow more coherent completions)
            repetition_penalty=1.2  # Slight penalty for repetition
        )

        # Decode the generated tokens into a string
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Trim the response to the last complete sentence
        trimmed_response = trim_to_complete_sentence(response)

        # Log the raw and trimmed responses for debugging
        print(f"Raw response from GPT-J: {response}")
        print(f"Trimmed response: {trimmed_response}")

        # Return the trimmed response as JSON
        return jsonify({"response": trimmed_response})

    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": str(e)})

# Start the Flask app on port 5001
if __name__ == '__main__':
    print("Starting Flask server on port 5001...")
    app.run(host='0.0.0.0', port=5001)  # Running on port 5001
