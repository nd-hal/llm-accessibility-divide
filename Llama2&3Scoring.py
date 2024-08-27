import os
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import replicate
import logging
import signal

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the environment variable for API 
os.environ['REPLICATE_API_TOKEN'] = "key was here"

# Load file
#df = pd.read_csv('/Users/koketch/Desktop/3-ShotFCE_scores2.csv', dtype=str)

#df = pd.read_csv('/Users/koketch/Desktop/IT 3-Shot/ASAP/ASAP_Set1_3-Shot.csv', dtype=str, encoding='ISO-8859-1')
#df = pd.read_csv('/Users/koketch/Desktop/IT 3-Shot/LLM Generated Text/LLaMA3-70B_generated_data.xlsx', dtype=str, encoding='ISO-8859-1', nrows=1800)
df = pd.read_excel('/Users/koketch/Desktop/IT 3-Shot/LLM Generated Text/0-Shot/Llama3.1-70B_generated_data.xlsx', dtype=str, nrows=1540)

# Convert all entries to string and strip them
df['0-Shot Rubric'] = df['0-Shot Rubric'].apply(lambda x: str(x).strip() if pd.notna(x) else "")

# Define a function to save the df
def save_dataframe(df, path):
    try:
        df.to_csv(path, index=False, float_format='%.2f')
        logging.info(f"Output file saved successfully at {path}")
    except Exception as e:
        logging.error(f"Failed to save the output file at {path}: {e}")

# Define a function to handle termination
def signal_handler(signal, frame):
    logging.info("Signal received, saving df before exit...")
    save_dataframe(df, '/Users/koketch/Desktop/IT 3-Shot/Llama3.1-70B_generated_dataTEMP.csv')
    exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Function to extract score from Llama 3 output
# def extract_score(response):
#     if isinstance(response, list):
#         response = ' '.join(response)
    
#     logging.debug(f"Extracted response for scoring: {response}")

#     # handle spaces around decimal point
#     score_match = re.search(r'Score\s*:\s*([\d,]+\s*\.?\s*\d*)', response, re.IGNORECASE)
#     if score_match:
#         # Remove commas and extra spaces from the captured score
#         score = score_match.group(1).replace(',', '').replace(' ', '')
#         logging.debug(f"Extracted score: {score}")
#         return score
#     else:
#         logging.error(f"No valid score extracted from the response: {response}")
#         return None

# # Example usage
# response = "The final Score: 18 . 5"
# extracted_score = extract_score(response)
# print("Extracted Score:", extracted_score)

# Function to extract score from Llama 2 output
def extract_score(response):
    # Normalize by reducing multiple spaces to a single space
    response = re.sub(r'\s+', ' ', response).strip()
    logging.debug(f"Normalized response for scoring: {response}")

    # handle erratic spaces within "Score", around, within digits, and decimals
    score_match = re.search(r'S *c *o *r *e *: *(\d+(?: \. \d+)?)', response, re.IGNORECASE)
    if score_match:
        # Extract score, remove all spaces from the captured digits to form the final score
        score_str = score_match.group(1)
        score_cleaned = re.sub(r'\s+', '', score_str)
        logging.debug(f"Extracted and cleaned score: {score_cleaned}")
        
        try:
            # Convert the cleaned score to float
            score_float = float(score_cleaned)
            return score_float
        except ValueError as e:
            logging.error(f"Failed to convert extracted score to float: {e}")
            return None
    else:
        logging.error(f"No valid score extracted from the response: {response}")
        return None

responses = ["Sc ore : 3 0 ", "Sc ore : 2 8 ", "Sc ore : 3 . 7 5"]#example from model output
for response in responses:
    print(f"Extracted score from '{response}': {extract_score(response)}")

for response in response:
    extracted_score = extract_score(response)
    print(f"Extracted Score from '{response}':", extracted_score)

def process_prompt(cell_content, index):
    instruction = "You are a virtual grading assistant. Directly provide a numeric score explicitly formatted as 'Score: [number]' followed by a detailed explanation."

    full_prompt = f"{instruction} Evaluate this essay: {cell_content}"

    try:
        response = replicate.run(
            #"meta/meta-llama-3-70b-instruct",
            "meta/llama-2-70b-chat",
            input={
                "prompt": full_prompt,
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.95,
                "stop_sequence": "\n",
                "repetition_penalty": 1.15
            }
        )
        if isinstance(response, list):
            response = ' '.join(response)
        response = response.strip()

        # Extract only numeric score
        score = extract_score(response)
        if score is not None:
            df.at[index, 'Llama2_0-Shot'] = float(score)
            df.at[index, 'Llama2_0-Shot Output'] = response
            logging.info(f"Processed prompt at index {index}, Score: {score}")
            return index, float(score), response
        else:
            raise ValueError("No valid score extracted.")
    except Exception as e:
        logging.error(f"Processing failed at index {index}: {e}")
        return index, None, "Failed to process prompt due to an error"


with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    count = 0
    start_index = 0 # Starting index
    end_index = 1540   # ending index

    for index, row in df.iloc[start_index:].iterrows():
        if index > end_index:
            break  # Stop processing if the current index exceeds 1537
        content = row['0-Shot Rubric']
        futures.append(executor.submit(process_prompt, content, index))

    for future in as_completed(futures):
        index, score, full_output = future.result()
        if index > end_index:
            continue  # Skip updating the df for indices beyond 1537
        df.at[index, 'Llama2_0-Shot'] = score
        df.at[index, 'Llama2_0-Shot Output'] = full_output
        count += 1
        if count % 500 == 0:
            temp_output_path = '/Users/koketch/Desktop/IT 3-Shot/LLM Generated Text/Llama3.1-70B_generated_dataTEMP.csv'
            save_dataframe(df, temp_output_path)
            logging.info(f"Temporary output file saved after processing {count} prompts at {temp_output_path}")

# Save the final results
final_output_path = '/Users/koketch/Desktop/Llama3.1-70B_generated_data.csv'
save_dataframe(df, final_output_path)