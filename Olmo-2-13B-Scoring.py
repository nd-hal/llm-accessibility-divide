import pandas as pd
import json
import time

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# Load and read the input files
input_file = "./Data/1-ShotHuman_generated_data.xlsx"
input_file = "./Data/1-ShotHuman_scored_by_Olmo2_13B.xlsx"
output_file = "./Data/1-ShotHuman_scored_by_Olmo2_13B_v2.xlsx"
temp_output_file = "./Data/1-ShotHuman_generated_dataTEMP.xlsx"
df = pd.read_excel(input_file)
df = df[df["Olmo2-13B"].str.startswith("!") | df["Olmo2-13B"].isna()]

assert(len(df["OneShotRubric"]) > 0)
print(len(df["OneShotRubric"]))

# Essays, instructions, and rubrics are in the 7th column
instructions_column = df["OneShotRubric"]

# Ensure the 'Llama3.1-70B_1-Shot' column is present and set to numeric type
if 'Olmo2-13B' not in df.columns:
    df['Olmo2-13B'] = pd.Series(dtype='float64')


# load the LLM model
llm = LLM(
    model="allenai/OLMo-2-1124-13B-Instruct",
    max_model_len=8192,
    rope_scaling={"factor":2,"rope_type":"linear"}
)

# Create a sampling params object.
guided_decoding_params = GuidedDecodingParams(regex=r"Score: \d+(\.*\d*)")
sampling_params = SamplingParams(
    temperature=0.7, 
    top_p=0.95,
    frequency_penalty= 1.0,
    max_tokens= 2000,
    guided_decoding=guided_decoding_params
)

system_prompt = "You are a virtual grading assistant. Directly provide a numeric score explicitly formatted as 'Score: [number]'."


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

conversations = [create_conversation(prompt) for prompt in instructions_column]

outputs = llm.chat(
    conversations,
    sampling_params=sampling_params,
    use_tqdm=True
)


def extract_scores(output):
    content = output.outputs[0].text.strip()
    #try:
    #    score_start = content.find("Score:") + len("Score:")
    #    score_str = content[score_start:].strip().split()[0]
    #    score = float(score_str)
    #except (ValueError, IndexError) as e:
    #    print(f"Error parsing score: {e}")
    #    score = None
    #except Exception as e:
    #    print(f"API call failed: {e}")
    #    score = None
    #return score
    return content

df["Olmo2-13B"] = [extract_scores(output) for output in outputs]

# Save the final df to the output file
df.to_excel(output_file, index=False)

print("Scores have been generated and saved to", output_file)

