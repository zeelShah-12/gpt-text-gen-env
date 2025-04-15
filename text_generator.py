import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import openai

# Set your OpenAI API key if using GPT-3
openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your actual API key

# For GPT-2 (using Hugging Face Transformers):
def generate_text_gpt2(prompt, max_length=50, temperature=0.7):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,  # Enable sampling
        num_return_sequences=1
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# For GPT-3 (using OpenAI API):
def generate_text_gpt3(prompt, max_tokens=50, temperature=0.7):
    response = openai.Completion.create(
        engine="text-davinci-003",  # or "gpt-3.5-turbo"
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    model_choice = input("Choose model (gpt2/gpt3): ").strip().lower()
    prompt = input("Enter your prompt: ")
    max_length = int(input("Enter max length (default 50): ") or 50)
    temperature = float(input("Enter temperature (default 0.7): ") or 0.7)

    if model_choice == "gpt2":
        generated_text = generate_text_gpt2(prompt, max_length, temperature)
    elif model_choice == "gpt3":
        generated_text = generate_text_gpt3(prompt, max_length, temperature)
    else:
        print("Invalid model choice. Please choose 'gpt2' or 'gpt3'.")
        exit(1)

    print("\nGenerated Text:\n")
    print(generated_text)