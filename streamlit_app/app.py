
import subprocess
import torch

# subprocess.call(['pip', 'install', 'uvicorn==0.17.6','fastapi==0.99.1','pydantic==1.10.10', 'requests==2.23.0','jinja2==3.1.2','python-multipart','numpy','pandas','setuptools-rust','accelerate'])
# subprocess.call(['pip', 'install', '--upgrade', 'transformers[torch]'])
import os

import streamlit as st
import requests
from transformers import GPT2LMHeadModel, GPT2Tokenizer ,GPT2Model
import joblib
import string



current_path = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(current_path, "gpt_tokenizer")
model_path = os.path.join(current_path, "gpt2_3epoch")
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path) # also try gpt2-medium
model = GPT2LMHeadModel.from_pretrained(model_path)

# model = joblib.load("custom_model.joblib")
# parameters = model.get_params()

# Print the parameters
# for param_name, param_value in parameters.items():
#     print(param_name, ":", param_value)
def clean_string(input_string):
    # Define a set of known characters
    known_characters = set(string.ascii_letters + string.digits + string.punctuation + string.whitespace)

    # Iterate over each character in the input string
    cleaned_string = ''.join(char for char in input_string if char in known_characters)
    last_full_stop_index = cleaned_string.rfind('.')

    # If a full stop is found, truncate the string to include only text up to that point
    if last_full_stop_index != -1:
        truncated_text = cleaned_string[:last_full_stop_index + 1]  # Include the full stop
    else:
        truncated_text = cleaned_string

    return truncated_text

def generate_text(model, tokenizer, prompt):
    device = "cpu"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.eos_token_id
    output = model.generate(input_ids, max_length=100, temperature=10 ,num_return_sequences=1, no_repeat_ngram_size=2, top_k=10, top_p=0.5, attention_mask=attention_mask,
                            pad_token_id=pad_token_id)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text[len(prompt):]

# def subject_gen_func(email):
#     device = "cpu"
#     prompt = email
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
#     attention_mask = torch.ones_like(input_ids)
#     pad_token_id = tokenizer.eos_token_id
#     output_ids = model.generate(input_ids, max_length=1024, num_return_sequences=1,attention_mask=attention_mask,
#             pad_token_id=pad_token_id)
#     generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return generated_text
    


def main():
    st.title("Question Answer on AI/ML Queries:")

    # Load the GPT-2 model and tokenizer

    # User input
    prompt_text = st.text_area("Enter Prompt Text")

    # Generate button
    if st.button("Generate answer"):
        if prompt_text:
            # Generate text using the loaded model
            generated_text = clean_string(generate_text(model=model, tokenizer=tokenizer, prompt=prompt_text))
            st.success(f"{generated_text}")
        else:
            st.warning("Please enter a prompt.")

if __name__ == "__main__":
    main()
