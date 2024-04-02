# provide L2R with all gold knowledge
import os
from tqdm import tqdm

from utils import *
from chain import L2R_Chain
from evaluate import *
from prompts.multiple_choice_1 import MULTIPLE_CHOICE_1_PROMPT_TEMPLATE

from env import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

model = L2R_Chain()


load_ratio = 1.0
load_num = int(load_ratio * len(load_mc_data('data/mc_task.json')))
output_path = 'output/gold_mc1_' + '{:.1f}'.format(load_ratio)


def enrich_knowledge():
    data = load_mc_data('data/mc_task.json')
    question_data = [D[0] for D in data]
    answer_data = [D[2] for D in data]

    model.enrich_gold_knowledge_from_truthfulqa(question_data, answer_data)
    model.save_knowledge('knowledge/truthfulqa_gold.txt')


def initialize():
    model.load_knowledge(['knowledge/truthfulqa_gold.txt'], load_part=load_num)
    model.init_knowledge_base()

    print('Initalization success...')


def mc1(data_path='data/mc_task.json'):
    data = load_mc_data(data_path, task='mc1')
    predictions = []
    for i, (question, candidate_answers, answer) in enumerate(tqdm(data)):
        file_path = output_path + '/' + str(i) +  '.json'
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


if __name__ == '__main__':
    # enrich_knowledge()
    initialize()

    # generation()

    data_path = 'data/mc_task.json'

    mc1(data_path)

    pred = gather_prediction(output_path, 'json')
    data = load_mc_data(data_path, 'mc1')
    gold = [D[2] for D in data]
    res = evaluate_multiple_choice_1(pred, gold)
    print(load_ratio, res)