"""
This script uses open ai to extend the concepts for query generation.
You need an open ai API for this.
"""
from argparse import ArgumentParser
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
import openai

def get_prompts_and_temperatures_financial(sentence) -> List[Tuple[str, str, float]]:
    few_shot_prompt = f"compelete with the words super set or synonym, but do not resue the exact same words, the word \"Super Set\" should not be in the response and response should have at least two words:\n" \
                      f"S&P 500=stock market index\n" \
                      f"Audi=car\n" \
                      f"Oil prices=petroleum prices\n" \
                      f"unemployment rate=unemployment percentage\n" \
                      f"iPhone sales=phone sales\n" \
                      f"Netflix shares=stock shares\n" \
                      f"President Trump=President\n" \
                      f"iPhone 11=iphone\n" \
                      f"Hong Kong=city\n" \
                      f"stake PEXA=Property Exchange Australia shares\n" \
                      f"{sentence}=\n"
    return few_shot_prompt, 1

def get_prompts_and_temperatures_clinical(sentence) -> List[Tuple[str, str, float]]:
    few_shot_prompt = f"compelete with the words super set or synonym, but do not resue the exact same words, the word \"Super Set\" should not be in the response and response should have at least two words:\n" \
                      f"ophthalmic solution=eye medication\n" \
                      f"Control group=treatment group\n" \
                      f"irinotecan hydrochloride= chemotherapy drug\n" \
                      f"monoclonal antibody=substitute antibodies\n" \
                      f"MRI scans=Magnetic resonance imaging\n" \
                      f"influenza H1N1 vaccine= flu vaccine\n" \
                      f"HAI antibody response=Influenza-specific antibody response\n" \
                      f"{sentence}=\n"
    return few_shot_prompt, 1



def get_args():
    args = ArgumentParser()

    args.add_argument("--data-type", type=str, default="finance",
                      help="either finance or clinical, used to select the correct prompt")
    args.add_argument("--input-file", type=str, default="../../data/finance/train_concepts_units_extended.csv",
                      help="a csv file that has a column `keyword` indicating the concepts")
    args.add_argument("--api-key", type=str, default="", help="Defines the output file for train")
    args.add_argument("--output-file", type=str, default="../../data/finance/concepts_extended.csv",
                      help="Path to output a csv file including all the extensions in a dictionary. ")

    return args.parse_args()

if __name__ == "__main__":
    args = get_args()
    openai.api_key=args.api_key

    df = pd.read_csv(args.input_file)
    continue_from = 0

    count = 0

    concepts = df["keyword"].unique()
    get_prompts_and_temperatures = get_prompts_and_temperatures_financial if args.data_type=="finance" else get_prompts_and_temperatures_clinical
    extention_data = []

    for idx, concept in enumerate(tqdm(concepts)):
        if idx < continue_from:
            continue
        else:
            count = count + 1
            prompt, temperature = get_prompts_and_temperatures(concept)

            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                stream=False,
                temperature=temperature,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            predictions = response["choices"][0]["text"]
            extention_data.append({"concept": concept, "extention": predictions})
        df_x = pd.DataFrame.from_dict(extention_data)
        df_x.to_csv(args.output_file, index=False)





