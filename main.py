import os
from tqdm import tqdm

from utils import *
from chain import L2R_Chain
from prompts.multiple_choice_1 import MULTIPLE_CHOICE_1_PROMPT_TEMPLATE
from prompts.multiple_choice_2 import MULTIPLE_CHOICE_2_PROMPT_TEMPLATE

from env import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

model = L2R_Chain()


def enrich_knowledge():
    data = load_truthfulqa('data/TruthfulQA.csv')
    # data = load_truthfulqa('data/TruthfulQA_demo.csv')
    question_data = [D['Question'] for D in data]

    model.enrich_knowledge_from_truthfulqa(question_data)
    model.save_knowledge('knowledge/truthfulqa_v0.txt')


def initialize():
    model.load_knowledge(['knowledge/truthfulqa_v0.txt'])
    model.init_knowledge_base()

    print('Initalization success...')
    # while True:
        # query = input('User: ')
        # answer = model.answer(query)
        # print('AI:', answer)


def generation():
    data = load_truthfulqa('data/TruthfulQA.csv')
    question_data = [D['Question'] for D in data]
    for i, q in enumerate(tqdm(question_data)):
        file_path = 'output/l2r_g' + str(i) +  '.json'
        if not os.path.exists(file_path):
            try:
                answer = model.answer(q, full_output=True)
                with open(file_path, 'w') as f:
                    json.dump(answer, f)
            except Exception as e:
                print(e)
                continue
        # break


def mc1(data_path='data/mc_task.json'):
    data = load_mc_data(data_path, task='mc1')
    predictions = []
    for i, (question, candidate_answers, answer) in enumerate(tqdm(data)):
        file_path = 'output/l2r_mc1/' + str(i) +  '.json'
        if not os.path.exists(file_path):
            model_input = MULTIPLE_CHOICE_1_PROMPT_TEMPLATE.format(question=question, candidate_answers=candidate_answers)
            try:
                model_output = model.answer(query=model_input, question=question, full_output=True)
                with open(file_path, 'w') as f:
                    json.dump(model_output, f)
            except Exception as e:
                print(e)
                print(model_output)
                continue
        # break
    return predictions


def mc2(data_path='data/mc_task.json'):
    data = load_mc_data(data_path, task='mc2')
    predictions = []
    for i, (question, candidate_answers, answer) in enumerate(tqdm(data)):
        file_path = 'output/l2r_mc2/' + str(i) +  '.json'
        if not os.path.exists(file_path):
            model_input = MULTIPLE_CHOICE_2_PROMPT_TEMPLATE.format(question=question, candidate_answers=candidate_answers)
            try:
                model_output = model.answer(query=model_input, question=question, full_output=True)
                with open(file_path, 'w') as f:
                    json.dump(model_output, f)
            except Exception as e:
                print(e)
                print(model_output)
                continue
        # break
    return predictions


if __name__ == '__main__':
    # enrich_knowledge()
    initialize()

    # generation()

    data_path = 'data/mc_task.json'
    # data_path = 'data/mc_task_random.json'

    mc1(data_path)

    mc2(data_path)