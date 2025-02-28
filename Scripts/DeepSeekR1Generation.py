
import pandas as pd
import replicate
import time
import os

# Set Replicate API token 
os.environ['REPLICATE_API_TOKEN'] = "Add your key here"

# Load the Excel file and clean column names
try:
    df = pd.read_excel('./Data/Data/Deepseek-R1_generated_data.xlsx')
    df.columns = [col.strip() for col in df.columns]
    print(f"Loaded Excel file with {len(df)} rows.")
except Exception as e:
    print(f"Failed to load the Excel file: {str(e)}")
    exit(1)

# Ensure the 'Response' column exists
if 'Response' not in df.columns:
    df['Response'] = pd.NA  # Initialize empty column
df['Response'] = df['Response'].astype('object')

# Define the starting row 
start_row = 373  # Change this to the row you want to start from

# Function to generate response
def generate_response(prompt):
    full_prompt = "Respond to the following question directly and concisely. No preamble, no explanations—only the response: " + prompt
    try:
        output = replicate.run(
            "deepseek-ai/deepseek-r1",
            input={
                "debug": False,
                "top_p": 0.95,
                "temperature": 0.7,
                "max_new_tokens": 2000,
                "min_new_tokens": 0,
                "repetition_penalty": 1.0,
                "batch_size": 8,
                "prompt": full_prompt
            }
        )

        # Replicate returns a list; join into a single string if needed
        if isinstance(output, list):
            response_text = "".join(output).strip()
        else:
            response_text = str(output).strip()

        print(f"Response received: {response_text[:100]}...")  # Show first 100 characters
        return response_text

    except Exception as e:
        print(f"Error generating response for prompt: {prompt[:50]}... Error: {str(e)}")
        return None  # Return None in case of failure

# Generate responses and update the dataframe
save_interval = 3 # Save every certain # of iterations
temp_file_prefix = './Data/Deepseek_generatedResponseTEMP_data.xlsx'

for index, row in df.iloc[start_row:].iterrows():  # Start from the specified row
    if pd.isna(row['Response']) or row['Response'] == "":  # Avoid overwriting existing responses
        df.at[index, 'Response'] = generate_response(row['Prompt'])
        time.sleep(10)  # Wait to prevent API rate limits
    
    # Save a temporary file every 'save_interval' prompts
    if (index - start_row + 1) % save_interval == 0:
        temp_file = f"{temp_file_prefix}_{index + 1}.xlsx"
        df.to_excel(temp_file, index=False)
        print(f"Intermediate file saved: {temp_file}")

# Save the final dataset
final_output = './Data/Deepseek-R1_generatedResponse_data.xlsx'
df.to_excel(final_output, index=False)
print(f"Final file saved: {final_output}")
