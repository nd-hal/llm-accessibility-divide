import pandas as pd
import time
import os
import random
import argparse
import openai
import replicate
import logging
import re
from llamaapi import LlamaAPI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    "Llama-3.1": {"model": "llama3.1-405b", "api": "llamaapi"},  # Also available via Replicate
    "Qwen-2.5": {"model": "Qwen/Qwen2.5-72B-Instruct", "api": "deepinfra"},
    "DeepSeek-R1": {"model": "deepseek-ai/deepseek-r1", "api": "replicate"},
    "GPT-4": {"model": "gpt-4", "api": "openai"},
    "GPT-4o": {"model": "gpt-4o", "api": "openai"},
    "GPT-3.5": {"model": "gpt-3.5-turbo", "api": "openai"}
}

# Load dataset
def load_data(input_file):
    try:
        df = pd.read_excel(input_file)
        df.columns = [col.strip() for col in df.columns]
        return df
    except Exception as e:
        logging.error(f"Error loading file {input_file}: {e}")
        exit(1)

# Extract numeric score from response
def extract_score(response):
    if not response:
        return None

    response = re.sub(r"\s+", " ", response).strip()  # Normalize spaces
    score_match = re.search(r"Score: *([-+]?\d+(?:\.\d+)?)", response, re.IGNORECASE)

    if score_match:
        try:
            return float(score_match.group(1))
        except ValueError:
            logging.error(f"Failed to convert extracted score to float: {score_match.group(1)}")
            return None
    else:
        logging.error(f"No valid score extracted from response: {response}")
        return None

# Generate scores using the specified model
def generate_score(essay, model_name, max_retries=3, initial_wait=2):
    model_info = MODEL_CONFIG.get(model_name)
    if not model_info:
        logging.error(f"Model {model_name} is not recognized.")
        return None

    api = model_info["api"]
    model = model_info["model"]

    prompt = "You are a virtual grading assistant. Directly provide a numeric score explicitly formatted as 'Score: [number]'."
    full_prompt = f"{prompt}\n{essay}"

    for attempt in range(max_retries):
        try:
            if api == "replicate":
                response = replicate_client.run(
                    model,
                    input={
                        "prompt": full_prompt,
                        "max_tokens": 1000,
                        "temperature": 0.7,
                        "top_p": 0.95,
                    }
                )
                response_text = ''.join(response).strip() if isinstance(response, list) else response.strip()

            elif api == "llamaapi":
                print(f"Sending request to Llama API for model {model}... (Attempt {attempt + 1})")
                
                llama_client = openai.OpenAI(api_key=LLAMA_API_KEY, base_url="https://api.llama-api.com/")
                response = llama_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": essay}
                    ],
                    max_tokens=1000,
                    temperature=0.7,
                    top_p=0.95
                )
                response_text = response.choices[0].message.content.strip()

            elif api == "deepinfra":
                response = deepinfra_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": essay}
                    ],
                    max_tokens=1000,
                    temperature=0.7,
                    top_p=0.95
                )
                response_text = response.choices[0].message.content.strip()

            elif api == "openai":
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": essay}
                    ],
                    max_tokens=1000,
                    temperature=0.7,
                    top_p=1.0
                )
                response_text = response.choices[0].message.content.strip()

            logging.info(f"{model_name} Scoring API Response: {response_text[:100]}...")
            return extract_score(response_text)

        except Exception as e:
            logging.error(f"Error generating score (Attempt {attempt + 1}): {e}")
            if "Concurrency conflict" in str(e):  # Handle Llama API 409 error
                wait_time = initial_wait * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                break  # Stop retrying if it's a different error

    return None  # Return None after max retries

# Process dataset and assign scores
def process_scoring(input_file, output_file, model_name, start_row=0, save_interval=500):
    df = load_data(input_file)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(generate_score, row['0-Shot Rubric'], model_name): index # change "0-Shot Rubric" to "1-Shot Rubric" for few-shot scoring
            for index, row in df.iloc[start_row:].iterrows()
        }

        for counter, future in enumerate(as_completed(futures), start=1):
            score = future.result()
            index = futures[future]
            if score is not None:
                df.at[index, f'{model_name}_0-Shot'] = score
                logging.info(f"Processed essay at index {index}, Score: {score}")

            if counter % save_interval == 0:
                temp_output_path = output_file.replace(".xlsx", f"_temp_{counter}.xlsx")
                df.to_excel(temp_output_path, index=False)
                logging.info(f"Temporary output saved at {temp_output_path}")

    df.to_excel(output_file, index=False)
    logging.info(f"Final dataset saved: {output_file}")

# Run script from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automated essay scoring using different LLMs.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output file.")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CONFIG.keys(), help="Model name to use for scoring.")
    parser.add_argument("--start_row", type=int, default=0, help="Row to start processing from.") #optional
    parser.add_argument("--save_interval", type=int, default=500, help="How often to save interim results.")#optional

    args = parser.parse_args()
    process_scoring(args.input_file, args.output_file, args.model, args.start_row, args.save_interval)
#Example usage: python Scripts/UnifiedScoring.py --input_file "./Data/Llama3_generated_data.xlsx" --output_file ./Data/Llama3_generatedTest.xlsx --model GPT-4 
