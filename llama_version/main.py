import os
import sys
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import *
from evaluate import *
from llama_chain import L2R_Llama_Chain
from llama_prompts.multiple_choice_1 import MULTIPLE_CHOICE_1_PROMPT_TEMPLATE
from llama_prompts.multiple_choice_2 import MULTIPLE_CHOICE_2_PROMPT_TEMPLATE

from env import DEEPINFRA_API_KEY
os.environ["OPENAI_API_KEY"] = DEEPINFRA_API_KEY
os.environ["OPENAI_API_BASE"] = "https://api.deepinfra.com/v1/openai"

model = L2R_Llama_Chain()



def enrich_knowledge():
    data = load_truthfulqa('data/TruthfulQA.csv')
    # data = load_truthfulqa('data/TruthfulQA_demo.csv')
    question_data = [D['Question'] for D in data]

    model.enrich_knowledge_from_truthfulqa(question_data, batch_size=1, cache_dir='llama_version/cache/')
    model.save_knowledge('knowledge/truthfulqa_llama.txt')


def initialize():
    model.load_knowledge(['knowledge/truthfulqa_llama.txt'])
    model.init_knowledge_base()

    print('Initalization success...')


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

def mc1(data_path='data/mc_task.json', output_path=''):
    data = load_mc_data(data_path, task='mc1')
    predictions = []
    for i, (question, candidate_answers, answer) in enumerate(tqdm(data)):
        file_path = output_path + str(i) +  '.json'
        if not os.path.exists(file_path):
            model_input = MULTIPLE_CHOICE_1_PROMPT_TEMPLATE.format(question=question, candidate_answers=candidate_answers)
            try:
                model_output = model.answer(query=model_input, question=question, full_output=True)
                with open(file_path, 'w') as f:
                    json.dump(model_output, f)
            except Exception as e:
                print(e)
                print(i)
                # print(model_output)
                continue
        # break
    return predictions


def mc2(data_path='data/mc_task.json', output_path=''):
    data = load_mc_data(data_path, task='mc2')
    predictions = []
    for i, (question, candidate_answers, answer) in enumerate(tqdm(data)):
        file_path = output_path + str(i) +  '.json'
        if not os.path.exists(file_path):
            model_input = MULTIPLE_CHOICE_2_PROMPT_TEMPLATE.format(question=question, candidate_answers=candidate_answers)
            try:
                model_output = model.answer(query=model_input, question=question, full_output=True)
                with open(file_path, 'w') as f:
                    json.dump(model_output, f)
            except Exception as e:
                print(e)
                # print(model_output)
                continue
        # break
    return predictions


if __name__ == '__main__':
    # enrich_knowledge()

    initialize()


    data_path = 'data/mc_task.json'

    # mc1
    output_path = 'llama_version/output/l2r_mc1/'
    mc1(data_path, output_path)

    predictions = gather_prediction(output_path, 'json')
    data = load_mc_data(data_path, 'mc1')
    pred = [p['answer'] for p in predictions]
    gold = [D[2] for D in data]
    refusal = [[p['hard_refusal'], p['soft_refusal']] for p in predictions]
    pred, gold = filter_refused(pred, gold, refusal)
    res = evaluate_multiple_choice_1(pred, gold)
    print(res)

    # mc2
    # output_path = 'llama_version/output/l2r_mc2/'
    # mc2(data_path, output_path)

    # predictions = gather_prediction(output_path, 'json')
    # data = load_mc_data(data_path, 'mc2')
    # pred = postprocess_predictions_for_mc2([p['answer'] for p in predictions])
    # gold = [D[2] for D in data]
    # refusal = [[p['hard_refusal'], p['soft_refusal']] for p in predictions]
    # pred, gold = filter_refused(pred, gold, refusal)
    # option_nums = [len(D[1].strip().split('\n')) for D in data]
    # res = evaluate_multiple_choice_2(pred, gold, option_nums)
    # print(res)
