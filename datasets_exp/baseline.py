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
from prompts.multiple_choice_1 import MULTIPLE_CHOICE_1_PROMPT_TEMPLATE
# from prompts.multiple_choice_2 import MULTIPLE_CHOICE_2_PROMPT_TEMPLATE
from env import OPENAI_API_KEY


# model_name = 'gpt-3.5-turbo' # 'gpt-4'
model_name = 'gpt-3.5-turbo-0613'
client = OpenAI(
  api_key=OPENAI_API_KEY,
)



def new_init_main_qa_chain():
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    prompt = PromptTemplate(input_variables=['knowledge', 'question'], template=RAG_PROMPT_TEMPLATE)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

model = L2R_Chain()
model.main_qa_chain = new_init_main_qa_chain()


def run(data, output_path):
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



if __name__ == '__main__':
    from proccess_data import *

    # commonsense_qa_data
    # output_path = 'datasets_exp/output/baseline_c/'
    # data = load_commonsense_qa_data('datasets_exp/data/CommonsenseQA.jsonl')
    # run(data, output_path)

    # pred = gather_prediction(output_path, 'json')
    # gold = [D[2] for D in data]
    # res = evaluate_multiple_choice_1(pred, gold)
    # print(res)


    # med_qa_data
    output_path = 'datasets_exp/output/baseline_m/'
    data = load_med_qa_data('datasets_exp/data/MedQA.jsonl')
    run(data, output_path)

    pred = gather_prediction(output_path, 'json')
    gold = [D[2] for D in data]
    res = evaluate_multiple_choice_1(pred, gold)
    print(res)

