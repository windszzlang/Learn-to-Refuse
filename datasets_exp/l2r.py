import os
import sys
from tqdm import tqdm
import openai

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import *
from evaluate import *
from proccess_data import *
from chain import L2R_Chain
from prompts.multiple_choice_1 import MULTIPLE_CHOICE_1_PROMPT_TEMPLATE
# from prompts.multiple_choice_2 import MULTIPLE_CHOICE_2_PROMPT_TEMPLATE

from env import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

model = L2R_Chain()
# model.retrieval_top_k = 4 # commonsense
# model.retrieval_top_k = 2 # medqa non-rag
model.retrieval_top_k = 4 # medqa rag


def enrich_knowledge(data, dataset_name):
    question_data = [D[0] for D in data]

    model.enrich_knowledge_from_truthfulqa(question_data, batch_size=1, cache_dir='datasets_exp/cache/' + dataset_name)
    model.save_knowledge('knowledge/' + str(dataset_name) + '.txt')


def initialize(dataset_name):
    model.load_knowledge(['knowledge/' + str(dataset_name) + '.txt'])
    model.init_knowledge_base()

    print('Initalization success...')
    # while True:
        # query = input('User: ')
        # answer = model.answer(query)
        # print('AI:', answer)


def run(data, output_path):
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
                print(model_output)
                continue
        # break
    return predictions


if __name__ == '__main__':


    ## commonsense_qa_data
    # dataset_name = 'commonsense_qa'
    # output_path = 'datasets_exp/output/l2r_c/'
    # data = load_commonsense_qa_data('datasets_exp/data/CommonsenseQA.jsonl')

    # # enrich knowledge and load
    # # enrich_knowledge(data, dataset_name)
    # initialize(dataset_name)

    # run(data, output_path)
    
    # predictions = gather_prediction(output_path, 'json')
    # pred = [p['answer'] for p in predictions]
    # gold = [D[2] for D in data]
    # refusal = [[p['hard_refusal'], p['soft_refusal']] for p in predictions]
    # pred, gold = filter_refused(pred, gold, refusal)
    # res = evaluate_multiple_choice_1(pred, gold)
    # print(res)

    ## med_qa_data
    dataset_name = 'med_qa'
    output_path = 'datasets_exp/output/l2r_m/'
    # output_path = 'datasets_exp/output/l2r_m_pure/'
    # output_path = 'datasets_exp/output/l2r_m_rag/'
    data = load_med_qa_data('datasets_exp/data/MedQA.jsonl')

    # enrich knowledge and load
    # enrich_knowledge(data, dataset_name)
    initialize(dataset_name) # non-rag
    # initialize('medrag_textbooks') # rag

    run(data, output_path)

    predictions = gather_prediction(output_path, 'json')
    pred = [p['answer'] for p in predictions]
    gold = [D[2] for D in data]
    refusal = [[p['hard_refusal'], p['soft_refusal']] for p in predictions]
    pred, gold = filter_refused(pred, gold, refusal)
    res = evaluate_multiple_choice_1(pred, gold)
    print(res)

