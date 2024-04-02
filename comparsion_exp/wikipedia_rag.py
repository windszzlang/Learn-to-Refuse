# experiments of gold data from wikipedia in RAG
import os
from tqdm import tqdm

from langchain import LLMChain, PromptTemplate
from langchain_openai import ChatOpenAI

from utils import *
from chain import L2R_Chain
from prompts.multiple_choice_1 import MULTIPLE_CHOICE_1_PROMPT_TEMPLATE
from prompts.multiple_choice_2 import MULTIPLE_CHOICE_2_PROMPT_TEMPLATE
from prompts.rag import RAG_PROMPT_TEMPLATE

from env import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


def new_init_main_qa_chain():
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    prompt = PromptTemplate(input_variables=['knowledge', 'question'], template=RAG_PROMPT_TEMPLATE)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

model = L2R_Chain()
model.main_qa_chain = new_init_main_qa_chain()


def initialize():
    # loading time ~1.5min
    model.load_knowledge_base('knowledge_base/wikipedia.pkl')
    print('Initalization success...')
    # while True:
    #     query = input('User: ')
    #     # answer = model.answer(query, wikipedia=True)
    #     answer = model.answer(query, full_output=True, wikipedia=True)
    #     print('AI:', answer)



def mc1():
    data = load_mc_data('data/mc_task.json', task='mc1')
    predictions = []
    for i, (question, candidate_answers, answer) in enumerate(tqdm(data)):
        file_path = 'output/wiki_rag_mc1/' + str(i) +  '.json'
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


def mc2():
    data = load_mc_data('data/mc_task.json', task='mc2')
    predictions = []
    for i, (question, candidate_answers, answer) in enumerate(tqdm(data)):
        file_path = 'output/wiki_rag_mc2/' + str(i) +  '.json'
        if not os.path.exists(file_path):
            model_input = MULTIPLE_CHOICE_2_PROMPT_TEMPLATE.format(question=question, candidate_answers=candidate_answers)
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

    mc1()

    mc2()