from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "EleutherAI/gpt-j-6B"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model loaded!")

# Test prompt
prompt = "Hello, how are you?"

# Tokenize input and create attention mask
input_ids = tokenizer.encode(prompt, return_tensors='pt')
attention_mask = torch.ones_like(input_ids)

# Generate response
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# Decode and print response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated response:", response)
