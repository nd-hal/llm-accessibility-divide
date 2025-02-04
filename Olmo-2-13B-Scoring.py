import pandas as pd
import json
import time

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# Load and read the input files
#input_file = "./Data/1-ShotHuman_generated_data.xlsx"
#input_file = "./Data/1-ShotHuman_scored_by_Olmo2_13B.xlsx"


datafolder = "./Data/"

def create_conversation(prompt):
    system_prompt = "You are a virtual grading assistant. Directly provide a numeric score explicitly formatted as 'Score: [number]'."

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


def grade_model_outputs(modelName, llm):

    input_file = f"./Data/{modelName}_generated_data.xlsx"
    output_file = f"./Data/{modelName}_graded_by_Olmo2_13B.csv"

    #output_file = "./Data/1-ShotHuman_scored_by_Olmo2_13B_v2.xlsx"
    #temp_output_file = "./Data/1-ShotHuman_generated_dataTEMP.xlsx"

    df = pd.read_excel(input_file)
    #df = df[df["Olmo2-13B"].str.startswith("!") | df["Olmo2-13B"].isna()]

    assert(len(df["1-Shot Rubric"]) > 0)
    print(len(df["1-Shot Rubric"]))

    # Essays, instructions, and rubrics are in the 7th column
    zeroShots = df["0-Shot Rubric"]
    oneShots = df["1-Shot Rubric"]

    outputDF = df[["Prompt ID","Prompt Type","Prompt Variation","Iteration"]]

    # Ensure the 'Llama3.1-70B_1-Shot' column is present and set to numeric type
    outputDF['Olmo2-13B-zeroshot'] = pd.Series(dtype='float64')
    outputDF['Olmo2-13B-oneshot'] = pd.Series(dtype='float64')

    

    # Create a sampling params object.
    guided_decoding_params = GuidedDecodingParams(regex=r"Score: \d+(\.*\d*)")
    sampling_params = SamplingParams(
        temperature=0.7, 
        top_p=0.95,
        frequency_penalty= 1.0,
        max_tokens= 2000,
        guided_decoding=guided_decoding_params
    )


    conversationsZS = [create_conversation(prompt) for prompt in zeroShots]

    outputs = llm.chat(
        conversationsZS,
        sampling_params=sampling_params,
        use_tqdm=True
    )

    outputDF['Olmo2-13B-zeroshot'] = [extract_scores(output) for output in outputs]


    conversationsOS = [create_conversation(prompt) for prompt in oneShots]

    outputs = llm.chat(
        conversationsOS,
        sampling_params=sampling_params,
        use_tqdm=True
    )

    outputDF['Olmo2-13B-oneshot'] = [extract_scores(output) for output in outputs]

    # Save the final df to the output file
    outputDF.to_csv(output_file, index=False)

    print("Scores have been generated and saved to", output_file)


# load the LLM model
# no rope scaling for now bc it doesnt work...
llm = LLM(
    model="allenai/OLMo-2-1124-13B-Instruct",
    #max_model_len=8192,
    #rope_scaling={"factor":2,"rope_type":"linear"}
)

modelnames = [
    #"GPT3.5", 
    "GPT4", 
    "GPT4o", 
    #"Llama2-70B", "Llama3-70B", "Llama3.1", "Qwen2.5-72B", "Deepseek-R1"
]


for m in modelnames:
    print(f"grading {m}")
    grade_model_outputs(m)


