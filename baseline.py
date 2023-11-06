import os
from tqdm import tqdm
import openai

from utils import *
from evaluate import *
from prompts.multiple_choice_1 import MULTIPLE_CHOICE_1_PROMPT_TEMPLATE
from prompts.multiple_choice_2 import MULTIPLE_CHOICE_2_PROMPT_TEMPLATE
from env import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY


model_name = 'gpt-3.5-turbo' # 'gpt-4'


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
                resp = openai.ChatCompletion.create(
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


def mc1(data_path):
    data = load_mc_data(data_path, task='mc1')
    predictions = []
    for i, (question, candidate_answers, answer) in enumerate(tqdm(data)):
        file_path = 'output/baseline_mc1/' + str(i) +  '.json'
        if not os.path.exists(file_path):
            model_input = MULTIPLE_CHOICE_1_PROMPT_TEMPLATE.format(question=question, candidate_answers=candidate_answers)
            model_input += '\nYour response:'
            try:
                resp = openai.ChatCompletion.create(
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


def mc2(data_path):
    data = load_mc_data(data_path, task='mc2')
    predictions = []
    for i, (question, candidate_answers, answer) in enumerate(tqdm(data)):
        file_path = 'output/baseline_mc2/' + str(i) +  '.json'
        if not os.path.exists(file_path):
            model_input = MULTIPLE_CHOICE_2_PROMPT_TEMPLATE.format(question=question, candidate_answers=candidate_answers)
            model_input += '\nYour response:'
            try:
                resp = openai.ChatCompletion.create(
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
    mc1(data_path)
    
    pred = gather_prediction('output/baseline_mc1', 'json')
    data = load_mc_data(data_path, 'mc1')
    gold = [D[2] for D in data]
    res = evaluate_multiple_choice_1(pred, gold)
    print(res)

    ## mc2
    mc2(data_path)
    
    pred = gather_prediction('output/baseline_mc2', 'json')
    data = load_mc_data(data_path, 'mc2')
    gold = [D[2] for D in data]
    option_nums = [len(D[1].strip().split('\n')) for D in data]
    res = evaluate_multiple_choice_2(pred, gold, option_nums)
    print(res)
