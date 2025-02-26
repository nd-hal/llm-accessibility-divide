import pandas as pd
import time
import openai 

# Initialize OpenAI Client for DeepInfra
client = openai.OpenAI(
    api_key="Add your key here",
    base_url="https://api.deepinfra.com/v1/openai"
)


# Load the input file
input_file = './Data/Qwen2.5_generated_data.xlsx'
output_file = '.Data/Qwen2.5_generatedResponse_data.xlsx'
temp_output_file = '.Data/Qwen2.5_generatedResponseTEMP_data.xlsx'

try:
    df = pd.read_excel(input_file)
    print(f"Loaded input file: {input_file}")
except Exception as e:
    print(f"Error loading input file: {e}")
    exit()

# Prompts are in column 5 (index 4)
try:
    prompts = df.iloc[:, 4]
    print(f"Found {len(prompts)} prompts.")
except Exception as e:
    print(f"Error reading prompts from file: {e}")
    exit()

# Ensures the 'Response' column exists and is set to string type
df['Response'] = df['Response'].astype(str)

# Function to generate response using DeepInfra
def generate_response(prompt):
    print(f"🔹 Generating response for: {prompt[:50]}...")  # print the first 50 chars of the prompt
    
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[
                {"role": "system", "content": "You are an AI text generation model assistant that provides responses to essay prompts. Your task is to respond to the provided prompt with a complete response."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=1.0
        )

        # Extract response content
        content = response.choices[0].message.content.strip()

        # Print the response
        print(f"Response received: {content[:100]}...")  # Show first 100 chars
        
        return content

    except Exception as e:
        print(f"Error generating response for prompt '{prompt[:50]}...': {e}")
        return f"Error: {e}"

# Generate responses
start_row = 0 # starting row
save_interval = 3  # Save the temporary file every # of prompts

for index, prompt in enumerate(prompts[start_row:], start=start_row):
    response = generate_response(prompt)
    df.at[index, 'Response'] = response
    time.sleep(1)  # Delay between API calls

    # Save a temporary output file every 'save_interval' runs
    if (index - start_row + 1) % save_interval == 0:
        df.to_excel(temp_output_file, index=False)
        print(f"Temporary output saved after {index - start_row + 1} prompts to {temp_output_file}")

# Save final responses
df.to_excel(output_file, index=False)
print(f"Responses have been generated and saved to {output_file}")
