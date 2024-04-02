# experiments of adjustment of gold data raito in the setting of L2R
import os
from tqdm import tqdm

from utils import *
from chain import L2R_Chain
from evaluate import *
from prompts.multiple_choice_1 import MULTIPLE_CHOICE_1_PROMPT_TEMPLATE

from env import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

model = L2R_Chain()


load_ratio = 0.75
load_num = int(load_ratio * len(load_mc_data('data/mc_task.json')))
print('load_num', load_num)
output_path = 'output/gold_l2r_mc1_' + '{:.2f}'.format(load_ratio)
if not os.path.exists(output_path):
    os.makedirs(output_path)


def enrich_knowledge():
    data = load_mc_data('data/mc_task.json', 'mc1')
    question_data = [D[0] for D in data]
    answer_data = [D[1].strip().split('\n')[D[2][0] - 1] for D in data]
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
    # initialize()

    # # generation()

    # data_path = 'data/mc_task.json'
    # mc1(data_path)

    # predictions = gather_prediction(output_path, 'json')
    # pred = [p['answer'] for p in predictions]
    # data = load_mc_data(data_path, 'mc1')
    # gold = [D[2] for D in data]
    # if output_path != 'wiki_rag':
    #     refusal = [[p['hard_refusal'], p['soft_refusal']] for p in predictions]
    #     pred, gold = filter_refused(pred, gold, refusal)

        
    # res = evaluate_multiple_choice_1(pred, gold)
    # print(load_ratio, res)




    ## test refusal success ratio
    data_path = 'data/mc_task.json'
    predictions = gather_prediction(output_path, 'json')
    pred = [p['answer'] for p in predictions]
    data = load_mc_data(data_path, 'mc1')
    gold = [D[2] for D in data]
    cnt = 0
    total = 0
    for i, (p, g) in enumerate(zip(pred, gold)):
        if predictions[i]['hard_refusal'] or predictions[i]['soft_refusal']:
            total += 1
        else:
            continue
        if p != 1:
            cnt += 1
    print(cnt / total)
