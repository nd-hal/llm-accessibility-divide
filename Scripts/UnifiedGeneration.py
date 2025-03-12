import pandas as pd
import time
import os
import argparse
import openai
import replicate
from llamaapi import LlamaAPI
from vllm import LLM, SamplingParams

# Load API Keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")

# Initialize API clients
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
deepinfra_client = openai.OpenAI(api_key=DEEPINFRA_API_KEY, base_url="https://api.deepinfra.com/v1/openai")
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
llama = LlamaAPI(LLAMA_API_KEY)  

# Model configuration
MODEL_CONFIG = {
    "Llama-3": {"model": "meta/meta-llama-3-70b-instruct", "api": "replicate"},
    "Llama-2": {"model": "meta/llama-2-70b-chat", "api": "replicate"},
    "Llama-3.1": {"model": "llama3.1-405b", "api": "llamaapi"}, # Also available via Replicate
    "Qwen-2.5": {"model": "Qwen/Qwen2.5-72B-Instruct", "api": "deepinfra"},
    "DeepSeek-R1": {"model": "deepseek-ai/deepseek-r1", "api": "replicate"}, 
    "Olmo-2-13B": {"model": "allenai/OLMo-2-1124-13B-Instruct", "api": "vllm"},
    "GPT-4": {"model": "gpt-4", "api": "openai"},
    "GPT-4o": {"model": "gpt-4o", "api": "openai"},
    "GPT-3.5": {"model": "gpt-3.5-turbo", "api": "openai"}
}

# Load dataset
def load_data(input_file):
    try:
        df = pd.read_excel(input_file)
        df.columns = [col.strip() for col in df.columns]
        if 'Response' not in df.columns:
            print("Adding 'Response' column to dataset.")
            df['Response'] = pd.NA
        df['Response'] = df['Response'].astype('object')
        return df
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        exit(1)

# Generate response based on model type
def generate_response(prompt, model_name):
    model_info = MODEL_CONFIG.get(model_name)
    if not model_info:
        print(f"Model {model_name} is not recognized.")
        return "Error: Model not found"

    api = model_info["api"]
    model = model_info["model"]

    print(f"Calling {api.upper()} API for model {model_name} with prompt: {prompt[:50]}...")

    try:
        #Handling Replicate API
        if api == "replicate":
            print(f"Sending request to replicate API for model {model}...")
            response = replicate_client.run(
                model,
                input={
                    "prompt": "You are an assistant that provides responses to essay prompts. Please provide a response to the following question without repeating the question:" + prompt,
                    "max_new_tokens": 4000,
                    "min_new_tokens": 0,
                    "repetition_penalty": 1.0,
                    "batch_size": 8,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "stream": False
                }
            )

            print(f"Raw replicate API response: {response}")  # Debugging
            return ''.join(response).strip() if isinstance(response, list) else response.strip()

        # Handling Llama API
        elif api == "llamaapi":
            print(f"Sending request to LlamaAPI for model {model}...")

            llama_client = openai.OpenAI(
                api_key=LLAMA_API_KEY, 
                base_url="https://api.llama-api.com/"
            )

            try:
                response = llama_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a llama assistant that provides responses to essay prompts. Your task is to respond to the provided prompt with a complete response."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.7,
                    top_p=0.95,
                    frequency_penalty=1.0,
                    stream=False
                )
                print(f"Raw LLaMA API response: {response}")  # Debugging
                
                # Extract message content
                if hasattr(response, "choices") and response.choices:
                    content = response.choices[0].message.content.strip()
                    print(f"Extracted LLaMA API response: {content[:100]}...")
                    return content

                print("Unexpected response structure:", response)
                return "Error: Unexpected response structure"

            except Exception as e:
                print(f"Error generating response for LLaMA API: {e}")
                return f"Error: {e}"

        # Handling DeepInfra API
        elif api == "deepinfra":
            print(f"Sending request to DeepInfra API for model {model}...")

            try:
                response = deepinfra_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an AI text generation model assistant that provides responses to essay prompts. Your task is to respond to the provided prompt with a complete response."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.7,
                    top_p=0.95,
                    frequency_penalty=1.0
                )
                print(f"DeepInfra API Response Received: {response}")  # Debugging
                
                if not response or not hasattr(response, "choices") or not response.choices:
                    print("DeepInfra API returned an empty response.")
                    return "Error: No response from model"
                
                result = response.choices[0].message.content.strip()
                print(f"DeepInfra Final Response: {result[:100]}...")
                return result

            except Exception as e:
                print(f"Error generating response from DeepInfra API: {e}")
                return f"Error: {e}"

        # Handling OpenAI API
        elif api == "openai":
            print(f"Sending request to OpenAI API for model {model}...")

            try:
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a virtual assistant that generates essay responses. Your task is provide a response to the prompt."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.7,
                    top_p=0.95,
                    frequency_penalty=1.0
                )

                result = response.choices[0].message.content.strip()
                print(f"OpenAI API response: {result[:100]}...")
                return result

            except Exception as e:
                print(f"Error generating response from OpenAI API: {e}")
                return f"Error: {e}"

    except Exception as e:
        print(f"Unexpected Error generating response: {e}")
        return f"Error: {e}"
    
# Process and generate responses
def process_generation(input_file, output_file, model_name, start_row=0, save_interval=10): #increase/decrease interval to your liking
    df = load_data(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")
    for index, row in df.iloc[start_row:].iterrows():
        if pd.isna(row['Response']):
            response = generate_response(row['Prompt'], model_name)
            df.at[index, 'Response'] = response
            print(f"Generated response for row {index}.")
            time.sleep(1)  # throttle requests to avoid hitting API limits
        if (index % save_interval) == 0:
            df.to_excel(output_file, index=False)
            print(f"Data saved after processing {index} rows.")
    df.to_excel(output_file, index=False)
    print("Final dataset saved.")

# Main function to run script from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text responses using LLMs.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIG.keys()), help="Model name to use.")
    parser.add_argument("--start_row", type=int, default=0, help="Row to start processing from.") #optional
    parser.add_argument("--save_interval", type=int, default=10, help="Interval to save intermediate results.") #optional
    args = parser.parse_args()
    process_generation(args.input_file, args.output_file, args.model, args.start_row, args.save_interval)

#Example usage: python Scripts/UnifiedGeneration.py --input_file "./Data/Llama3_generated_data.xlsx" --output_file ./Data/Llama3_generatedTest.xlsx --model Llama-3 
