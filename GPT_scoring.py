import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the API key
api_key = 'key was here'
openai.api_key = api_key

# Define start and end rows
start_row = 2
end_row = 1540  

# Load the file and slice the df
df = pd.read_excel('/Users/koketch/Desktop/IT 3-Shot/LLM Generated Text/1-Shot/Llama3.1-70B_generated_data.xlsx', dtype=str)
df = df.iloc[start_row:end_row]
logging.info(f"Loaded DataFrame with {df.shape[0]} rows from row {start_row} to {end_row}.")

def extract_score(response):
    # Extract score 
    if response:
        try:
            return response.split('Score: ')[1].split()[0]
        except IndexError:
            logging.error("Score format not found in response.")
            return None
    return None

def process_prompt(essay):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o", #specify model
            messages=[
                {"role": "system", "content": "You are a virtual grading assistant. Directly provide a numeric score explicitly formatted as 'Score: [number]'"},
                {"role": "user", "content": essay}
            ],
            max_tokens=1000,
            temperature=0.7,
            top_p=1.0,
            n=1,
            stop=None
        )
        return extract_score(response['choices'][0]['message']['content'].strip())
    except Exception as e:
        logging.error(f"Failed to process essay: {e}")
        return None

# ThreadPoolExecutor to process essays in parallel
results = {}
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(process_prompt, row['1-Shot Rubric']): index for index, row in df.iterrows()}
    counter = 0
    for future in as_completed(futures):
        score = future.result()
        index = futures[future]
        if score:
            results[index] = score
            logging.info(f"Processed essay at index {index}, Score: {score}")
            counter += 1
        if counter % 500 == 0:
            temp_output_path = f'/Users/koketch/Desktop/IT 3-Shot/LLM Generated Text/3.1temp_output_{counter // 500}.csv'
            df.loc[list(results.keys()), 'GPT4o_1-Shot'] = pd.Series(results)
            df.to_csv(temp_output_path, index=False)
            logging.info(f"Temporary output file saved after processing {counter} essays at {temp_output_path}")

# Update df with all results
df.loc[list(results.keys()), 'GPT4o-1Shot'] = pd.Series(results)

# Save the final results
output_path = '/Users/koketch/Desktop//IT 3-Shot/LLM Generated Text/1-ShotLlama3.1_generated_dataFINALE.csv'
try:
    df.to_csv(output_path, index=False)
    logging.info(f"Output file saved successfully at {output_path}")
except Exception as e:
    logging.error(f"Failed to save the output file: {e}")
# Update # Add a small change
