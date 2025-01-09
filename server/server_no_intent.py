from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS to handle cross-origin requests
import torch
from transformers import GPTJForCausalLM, AutoTokenizer

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load GPT-J model and tokenizer once when the server starts
print("Loading the GPT-J 6B model and tokenizer...")
try:
    model_name = "EleutherAI/gpt-j-6B"
    model = GPTJForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define the route for generating a response
@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        # Get user input from the request
        data = request.json
        user_input = data.get("message", "")

        # Tokenize the input
        input_ids = tokenizer.encode(user_input, return_tensors='pt')

        # Create an attention mask (non-zero values indicate the presence of input tokens)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

        # Generate a response from GPT-J with tuned parameters
        output = model.generate(
            input_ids,
            attention_mask=attention_mask, 
            max_length=100,  # Increased max length for more complete responses
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,  # Slightly higher temperature for creativity
            top_k=50,         # Filtering top 50 options
            top_p=0.9,        # Nucleus sampling for diversity
            repetition_penalty=1.1  # Slight repetition penalty
        )

        # Decode the generated tokens into a string
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Return the response as JSON without further modification
        return jsonify({"response": response})

    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": str(e)})

# Start the Flask app
if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000)
