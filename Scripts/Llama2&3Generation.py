import pandas as pd
import replicate
import time
import os


# Set API token through environment variable
os.environ['REPLICATE_API_TOKEN'] = "key was here"

# Initialize the Replicate client
client = replicate.Client()
# Load the Excel file and clean column names
try:
    df = pd.read_excel('./Data/Llama-70B_generated_data.xlsx')
    df.columns = [col.strip() for col in df.columns]
except Exception as e:
    print(f"Failed to load the Excel file: {str(e)}")
    exit(1)

# Ensure the 'Response' column exists and is of type object (to handle strings properly)
if 'Response' not in df.columns:
    df['Response'] = pd.NA
df['Response'] = df['Response'].astype('object')

# Function to generate response 
def generate_response(prompt):
    full_prompt = "Please provide a response to the following question without repeating it: " + prompt
    try:
        output = replicate.run(
            "meta/llama-2-70b-chat",
            input={"debug": False,
                "top_p": 0.95,
                "temperature": 0.7,
                "max_new_tokens": 1000,
                "min_new_tokens": 0,
                "repetition_penalty": 1.15,
                "batch_size": 8,
                "prompt": full_prompt}
        )
        return output[0] 
    except Exception as e:
        print(f"Error generating response for prompt: {prompt}. Error: {str(e)}")
        return None  # Return None in case of failure

# Generate responses and update the df with a delay
for index, row in df.iterrows():
    df.at[index, 'Response'] = generate_response(row['Prompt'])
    time.sleep(10)  # Wait for 10 seconds before processing the next prompt
    
    # Save the df every 20 iterations
    if (index + 1) % 20 == 0:
        df.to_excel(f'./Data/TEMPLlama2-70B_generated_data_{index + 1}.xlsx', index=False)
        #print(f"Intermediate file saved at index {index + 1}")

# Save the final version of the df
df.to_excel('./Data/UpdatedLlama-70B_generated_data_final.xlsx', index=False)
print("Final file saved")
