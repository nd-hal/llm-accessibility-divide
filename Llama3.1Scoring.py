import pandas as pd
import json
import time
from llamaapi import LlamaAPI

# Initialize API Key
llama = LlamaAPI("Add your key here")

# Load and read the input files
output_file = ".Data/OlmobyLlama3.1_scored.xlsx"
temp_output_file = ".Data/3.1__dataTEMPSCORE.xlsx"
input_file = '.Data/Olmo2-13B_generated_data.xlsx'
df = pd.read_excel(input_file)

# Essays, instructions, and rubrics are in the 7th column
instructions_column = df.iloc[:, 6]

# Ensure the 'Llama3.1-70B_1-Shot' column is present and set to numeric type
if 'Llama3.1_0-Shot' not in df.columns:
    df['Llama3.1_0-Shot'] = pd.Series(dtype='float64')

# Function to generate score 
def generate_score(instruction):
    api_request_json = {
        "model": "llama3.1-405b",  # Specify the model 
        "messages": [
            {"role": "system", "content": "You are a virtual llama grading assistant. Directly provide a numeric score explicitly formatted as 'Score: [number]'."},
            {"role": "user", "content": instruction},
        ],
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.95,
        "frequency_penalty": 1.0,
        "stream": False,
    }
    try:
        response = llama.run(api_request_json)
        response_data = response.json()
        content = response_data['choices'][0]['message']['content'].strip()
        print(f"Response content: {content}")  # Debug print to see the raw response
        
        # Extract the numeric score from the response content
        score_start = content.find("Score:") + len("Score:")
        score_str = content[score_start:].strip().split()[0]
        score = float(score_str)
    except (ValueError, IndexError) as e:
        print(f"Error parsing score: {e}")
        score = None
    except Exception as e:
        print(f"API call failed: {e}")
        score = None
    return score

# Generate scores 
start_row = 0  # starting row
end_row = 1540  # ending row
save_interval = 500  # Save the temporary file every # of prompts

for index, instruction in enumerate(instructions_column[start_row:end_row], start=start_row):
    score = generate_score(instruction)
    print(f"Score for index {index}: {score}")  # Debug print to see the parsed score
    # Add the score to the df immediately to save progress
    df.at[index, 'Llama3.1_0-Shot'] = score
    print(f"Updated df at index {index} with score {score}")  # Debug print to confirm df update
    # Delay between API calls
    time.sleep(1) 

    # Save a temporary output file every 'save_interval' runs
    if (index - start_row + 1) % save_interval == 0:
        df.to_excel(temp_output_file, index=False)
        print(f"Temporary output saved after {index - start_row + 1} prompts to", temp_output_file)

# Save the final df to the output file
df.to_excel(output_file, index=False)

#print("Scores have been generated and saved to", output_file)
