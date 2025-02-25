import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up DeepInfra OpenAI client
openai = OpenAI(
    api_key="5Gocw9C0FXamPPdMVj3bEiyKQMehXJbO",  
    base_url="https://api.deepinfra.com/v1/openai"
)

# Define start and end rows
start_row = 0
end_row = 15445  

# Load the file and slice the DataFrame
df = pd.read_excel('/afs/crc.nd.edu/user/k/koketch/Human_generated_dat\
a.xlsx')
df = df.iloc[start_row:end_row]
logging.info(f"Loaded DataFrame with {df.shape[0]} rows from row {start_row} to {end_row}.")

def extract_score(response):
    """Extracts the numeric score from DeepInfra response."""
    if response:
        try:
            return response.split('Score: ')[1].split()[0]
        except IndexError:
            logging.error("Score format not found in response.")
            return None
    return None

def process_prompt(essay):
    """Sends a request to DeepInfra model and extracts the score."""
    try:
        chat_completion = openai.chat.completions.create(
             model="Qwen/Qwen2.5-72B-Instruct",  
            messages=[
                {"role": "system", "content": "You are a virtual grading assistant. Directly provide a numeric score explicitly formatted as 'Score: [number]'"},
                {"role": "user", "content": essay}
            ],
            max_tokens=1000,
            temperature=0.7,
            top_p=1.0,
            n=1
        )

        return extract_score(chat_completion.choices[0].message.content.strip())

    except Exception as e:
        logging.error(f"Failed to process essay: {e}")
        return None

# Use ThreadPoolExecutor to process essays in parallel
results = {}
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(process_prompt, row['0-Shot Rubric']): index for index, row in df.iterrows()}
    counter = 0

    for future in as_completed(futures):
        score = future.result()
        index = futures[future]
        if score:
            results[index] = score
            logging.info(f"Processed essay at index {index}, Score: {score}")
            counter += 1

        # Save temporary results every 1500 essays
        if counter % 2000 == 0:
            temp_output_path = f'/afs/crc.nd.edu/user/k/koketch/QwenHumanTEMP_generated_dat\
a{counter // 500}.xlsx'
            df.loc[list(results.keys()), 'Qwen_0-Shot'] = pd.Series(results)
            df.to_csv(temp_output_path, index=False)
            logging.info(f"Temporary output file saved after processing {counter} essays at {temp_output_path}")

# Update DataFrame with all results
df.loc[list(results.keys()), 'Qwen-0-Shot'] = pd.Series(results)

# Save final results
output_path = '/afs/crc.nd.edu/user/k/koketch/QwenHuman_generated_dat\
a.xlsx'
try:
    df.to_csv(output_path, index=False)
    logging.info(f"Output file saved successfully at {output_path}")
except Exception as e:
    logging.error(f"Failed to save the output file: {e}")
