import pandas as pd
import json
import time
from llamaapi import LlamaAPI

# Initialize Llama API Key
llama = LlamaAPI("Key was here")

# Load/read the input file
input_file = '.Data/Llama3.1-70B_generated_data.xlsx'
output_file = '.Data/Llama3.1-70B_generatedResponse_data.xlsx'
temp_output_file = '.Data/Llama3.1-70B_generatedResponseTEMP_data.xlsx'
df = pd.read_excel(input_file)

# Prompts are in column 5(index 4)
prompts = df.iloc[:, 4]

# Ensure the 'Response' column set to string type. Throwing an error if not specified
df['Response'] = df['Response'].astype(str)

# Function to generate response 
def generate_response(prompt):
    api_request_json = {
        "model": "llama3.1-70b",
        "messages": [
            {"role": "system", "content": "You are a llama assistant that provides responses to essay prompts. Your task is to respond to the provided prompt with a complete response."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2000,  
        "temperature": 0.7,
        "top_p": 0.95,
        "frequency_penalty": 1.0,
        "stream": False,
    }
    try:
        response = llama.run(api_request_json)
        response_data = response.json()
        # Ensures the response contains the expected structure
        if 'choices' in response_data and len(response_data['choices']) > 0 and 'message' in response_data['choices'][0] and 'content' in response_data['choices'][0]['message']:
            content = response_data['choices'][0]['message']['content'].strip()
            # Check if the response ends abruptly and needs continuation
            if content.endswith(('...', 'incomplete', 'more', 'continued', 'in progress')):
                continuation = generate_response(prompt + " [Continue the response]")
                content += continuation
            return content
        else:
            print(f"Unexpected response structure: {response_data}")
            return "Error: Unexpected response structure"
    except Exception as e:
        print(f"Error generating response for prompt '{prompt}': {e}")
        return f"Error: {e}"

# Generate responses
responses = []
start_row = 2  # starting row
save_interval = 10  # Save the temporary file every 10 prompts

for index, prompt in enumerate(prompts[start_row:], start=start_row):
    response = generate_response(prompt)
    responses.append(response)
    # Add the response to the df immediately to save progress
    df.at[index, 'Response'] = response
    # Delay between API calls
    time.sleep(1)  

    # Save a temporary output file every 'save_interval' runs
    if (index - start_row + 1) % save_interval == 0:
        df.to_excel(temp_output_file, index=False)
        print(f"Temporary output saved after {index - start_row + 1} prompts to", temp_output_file)

# Save the final df to the output file
df.to_excel(output_file, index=False)

print("Responses have been generated and saved to", output_file)
