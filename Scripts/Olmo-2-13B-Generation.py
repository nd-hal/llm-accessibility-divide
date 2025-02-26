import pandas as pd
import json
import time

from vllm import LLM, SamplingParams

# Load/read the input file
input_file = './Data/llm_prompts.xlsx'
output_file = './Data/Olmo2-13B_generatedResponse_data.xlsx'
temp_output_file = './Data/Olmo2-13B_generatedResponseTEMP_data.xlsx'
df = pd.read_excel(input_file)

# Ensure the 'Response' column set to string type. Throwing an error if not specified
df["Response"] = ""
df['Response'] = df['Response'].astype(str)

# load the LLM model
llm = LLM(
    model="allenai/OLMo-2-1124-13B-Instruct",
    max_model_len=4096,
)

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.7, 
    top_p=0.95,
    frequency_penalty= 1.0,
    max_tokens= 2000,
)

system_prompt = "You are OLMo 2, a helpful and harmless AI Assistant built by the Allen Institute for AI that provides responses to essay prompts. Your task is to respond to the provided prompt with a complete response."

# Prompts are in column 5(index 4)
prompts = df["Prompt"]

def create_conversation(prompt):
    msg = [
                {
                "role": "system",
                "content": system_prompt
                },
                {"role": "user",
                 "content": prompt
                }
    ]
    return msg

conversations = [create_conversation(prompt) for prompt in prompts]

print("running inference...")

outputs = llm.chat(
    conversations,
    sampling_params=sampling_params,
    use_tqdm=True
)


print("parsing responses")

responses = [output.outputs[0].text.strip() for output in outputs]

df["Response"] = responses


# Save the final df to the output file
df.to_excel(output_file, index=False)

print("Responses have been generated and saved to", output_file)
