# experiments of adjustment of gold data raito in the setting of RAG
import os
from tqdm import tqdm

from langchain import LLMChain, PromptTemplate
from langchain_openai import ChatOpenAI

from utils import *
from evaluate import *
from chain import L2R_Chain
from prompts.multiple_choice_1 import MULTIPLE_CHOICE_1_PROMPT_TEMPLATE
from prompts.multiple_choice_2 import MULTIPLE_CHOICE_2_PROMPT_TEMPLATE
from prompts.rag import RAG_PROMPT_TEMPLATE

from env import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


def new_init_main_qa_chain():
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0)
    prompt = PromptTemplate(input_variables=['knowledge', 'question'], template=RAG_PROMPT_TEMPLATE)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

model = L2R_Chain()
model.main_qa_chain = new_init_main_qa_chain()


load_ratio = 0.25
load_num = int(load_ratio * len(load_mc_data('data/mc_task.json')))
output_path = 'output/gold_rag_mc1_' + '{:.2f}'.format(load_ratio)
if not os.path.exists(output_path):
    os.makedirs(output_path)



def initialize():
    model.load_knowledge(['knowledge/truthfulqa_gold.txt'], load_part=load_num)
    model.init_knowledge_base()

    print('Initalization success...')


def mc1(data_path):
    data = load_mc_data(data_path, task='mc1')
    predictions = []
    for i, (question, candidate_answers, answer) in enumerate(tqdm(data)):
        file_path = output_path + '/' + str(i) +  '.json'
        if not os.path.exists(file_path):
            model_input = MULTIPLE_CHOICE_1_PROMPT_TEMPLATE.format(question=question, candidate_answers=candidate_answers)
            try:
                model_output = model.answer(query=model_input, question=question, full_output=True, wikipedia=True, original_output=True)
                with open(file_path, 'w') as f:
                    json.dump(model_output, f)
            except Exception as e:
                print(e)
                continue
        # break
    return predictions



if __name__ == '__main__':

    initialize()

    data_path = 'data/mc_task.json'
    mc1(data_path)

    predictions = gather_prediction(output_path, 'json')
    pred = [p['answer'] for p in predictions]
    data = load_mc_data(data_path, 'mc1')
    gold = [D[2] for D in data]
    if 'rag' not in output_path:
        refusal = [[p['hard_refusal'], p['soft_refusal']] for p in predictions]
        pred, gold = filter_refused(pred, gold, refusal)
    res = evaluate_multiple_choice_1(pred, gold)
    print(load_ratio, res)