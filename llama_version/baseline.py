import os
import sys
from tqdm import tqdm
import openai
from openai import OpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import *
from evaluate import *
from llama_prompts.multiple_choice_1 import MULTIPLE_CHOICE_1_PROMPT_TEMPLATE
from llama_prompts.multiple_choice_2 import MULTIPLE_CHOICE_2_PROMPT_TEMPLATE

from env import DEEPINFRA_API_KEY
# os.environ["OPENAI_API_KEY"] = DEEPINFRA_API_KEY
# os.environ["OPENAI_API_BASE"] = "https://api.deepinfra.com/v1/openai"


model_name = 'meta-llama/Llama-2-70b-chat-hf' # 'meta-llama/Llama-2-7b-chat-hf'
client = OpenAI(
  api_key=DEEPINFRA_API_KEY,
  base_url="https://api.deepinfra.com/v1/openai"
)

def generation():
    # data = load_truthfulqa('data/TruthfulQA_demo.csv')
    data = load_truthfulqa('data/TruthfulQA.csv')
    question_data = [D['Question'] for D in data]
    predictions = []

    for i, D in enumerate(tqdm(question_data)):
        file_path = 'output/baseline_g/' + str(i) +  '.txt'
        if not os.path.exists(file_path):
            model_input = D
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {'role': 'user', 'content': model_input}
                    ],
                    temperature=0
                )
                model_output = resp.choices[0].message.content
                predictions.append(model_output)

                # print(model_input)
                # print(model_output)

                with open(file_path, 'w') as f:
                    f.write(model_output)
            except Exception as e:
                print(e)
                continue
    return predictions


def mc1(data_path, output_path):
    data = load_mc_data(data_path, task='mc1')
    predictions = []
    for i, (question, candidate_answers, answer) in enumerate(tqdm(data)):
        file_path = output_path + str(i) +  '.json'
        if not os.path.exists(file_path):
            model_input = MULTIPLE_CHOICE_1_PROMPT_TEMPLATE.format(question=question, candidate_answers=candidate_answers)
            model_input += '\nYour response:'
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {'role': 'user', 'content': model_input}
                    ],
                    temperature=0
                )
                model_output = resp.choices[0].message.content
                predictions.append(model_output)

                # print(model_input)
                # print(model_output)

                with open(file_path, 'w') as f:
                    f.write(model_output)
            except Exception as e:
                print(e)
                continue
        # break
    return predictions


def mc2(data_path, output_path):
    data = load_mc_data(data_path, task='mc2')
    predictions = []
    for i, (question, candidate_answers, answer) in enumerate(tqdm(data)):
        file_path = output_path + str(i) +  '.json'
        if not os.path.exists(file_path):
            model_input = MULTIPLE_CHOICE_2_PROMPT_TEMPLATE.format(question=question, candidate_answers=candidate_answers)
            model_input += '\nYour response:'
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {'role': 'user', 'content': model_input}
                    ],
                    temperature=0
                )
                model_output = resp.choices[0].message.content
                predictions.append(model_output)

                # print(model_input)
                # print(model_output)

                with open(file_path, 'w') as f:
                    f.write(model_output)
            except Exception as e:
                print(e)
                continue
        # break
    return predictions




if __name__ == '__main__':
    ## generation
    # generation()

    # prediction = gather_prediction('output/baseline_g', 'txt')
    # data = load_truthfulqa('data/TruthfulQA.csv')
    # true_answer_data = [D['Correct Answers'] for D in data]
    # false_answer_data = [D['Incorrect Answers'] for D in data]
    # res = evaluate_generation(prediction, true_answer_data, false_answer_data)
    # print(res)

    data_path = 'data/mc_task.json'
    # data_path = 'data/mc_task_random.json'

    ## mc1
    mc1(data_path, 'llama_version/output/baseline_mc1/')
    
    pred = gather_prediction('llama_version/output/baseline_mc1/', 'txt')
    pred = [extract_mc1_answer_from_llama_output(p) for p in pred]
    data = load_mc_data(data_path, 'mc1')
    gold = [D[2] for D in data]
    res = evaluate_multiple_choice_1(pred, gold)
    print(res)

    ## mc2
    mc2(data_path, 'llama_version/output/baseline_mc2/')
    
    pred = gather_prediction('llama_version/output/baseline_mc2/', 'txt')
    pred = [extract_mc2_answer_from_llama_output(p) for p in pred]
    data = load_mc_data(data_path, 'mc2')
    gold = [D[2] for D in data]
    option_nums = [len(D[1].strip().split('\n')) for D in data]
    res = evaluate_multiple_choice_2(pred, gold, option_nums)
    print(res)
