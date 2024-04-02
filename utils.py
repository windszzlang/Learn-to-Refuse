import os
import json
import re
import pickle
import random
import pandas as pd
import torch
from tqdm import tqdm
import faiss
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
# from datasets import load_dataset



def load_truthfulqa(data_path='data/TruthfulQA_demo.csv'):
    df = pd.read_csv(data_path)
    data = []
    for index, row in df.iterrows():
        D = {
            "Question": row["Question"],
            "Best Answer": row["Best Answer"],
            "Correct Answers": row["Correct Answers"],
            "Incorrect Answers": row["Incorrect Answers"]
        }
        data.append(D)
    return data


def load_mc_data(data_path='data/mc_task.json', task='mc1'):
    if task == 'mc1':
        key = 'mc1_targets'
    elif task == 'mc2':
        key = 'mc2_targets'
    else:
        raise ValueError

    with open(data_path) as f:
        data = json.load(f)
    questions = []
    candidate_answers = []
    answers = []
    for D in data:
        questions.append(D['question'])
        candidate_answer = ''
        answer = []
        answer_list = list(D[key].items())
        for i, (ans, is_true) in enumerate(answer_list):
            candidate_answer += str(i + 1) + ': ' + ans + '\n'
            if is_true:
                answer.append(i + 1)
        candidate_answers.append(candidate_answer)
        answers.append(answer)
    
    new_data = []
    for q, c, a in zip(questions, candidate_answers, answers):
        new_data.append([q, c, a])
    # print(new_data)
    return new_data


def gather_prediction(data_path, data_format='txt'):
    predictions = []
    filenames = os.listdir(data_path)
    filenames.sort(key=lambda x: int(x.split('.')[0])) 
    for file in filenames:
        file_path = os.path.join(data_path, file)
        # print(file_path)
        if os.path.isfile(file_path):
            # print(file)
            with open(file_path) as f:
                if data_format == 'json':
                    D = json.load(f)
                elif data_format == 'txt':
                    D = f.read().strip()
        predictions.append(D)
    return predictions


def save_knowledge_base(knowledge_base, knowledge_base_path):
    with open(knowledge_base_path, 'wb') as f:
        pickle.dump(knowledge_base, f)


def merge_knowledge_base(split_path, knowledge_base_path, split=10):
    wikipedia_knowledge_base = None
    for s in range(split):
        split_name = split_path + 'wikipedia_' + str(s) +  '.pkl'
        with open(split_name, 'rb') as f:
            tmp_kb = pickle.load(f)
            if wikipedia_knowledge_base == None:
                wikipedia_knowledge_base = tmp_kb
            else:
                wikipedia_knowledge_base.merge_from(tmp_kb)
    save_knowledge_base(wikipedia_knowledge_base, knowledge_base_path)


def load_wikipedia_knowledge_base(data_files=None, split=10, save_path='knowledge_base/wikipedia.pkl'):
    # https://huggingface.co/datasets/wikipedia
    if data_files:
        wikipedia_data = load_dataset("wikipedia", "20220301.en", data_files=data_files, beam_runner='DirectRunner')
    else:
        wikipedia_data = load_dataset("wikipedia", "20220301.en", beam_runner='DirectRunner')
    
    vectorstore = None
    embeddings = HuggingFaceEmbeddings( # too expensive if use gpt-embedding
        # model_name='sentence-transformers/all-MiniLM-L12-v2',
        model_name='sentence-transformers/all-mpnet-base-v2',
        model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    total_num = len(wikipedia_data['train'])
    s = 0
    for i in tqdm(range(total_num)):
        # store files by files and reset
        if i >= total_num / split * (s + 1): # i: [0, total_num-1]
            split_name = save_path.split('.')[0] + '_' + str(s) +  '.pkl'
            save_knowledge_base(vectorstore, split_name)
            s += 1
            del vectorstore
            vectorstore = None

        text = wikipedia_data['train'][i]['text'].split('\n')[0]
        document = Document(page_content=text, metadata={"source": "local", "confidence": 1.0})
        if vectorstore == None:
            vectorstore = FAISS.from_documents([document], embeddings)
        else:
            vectorstore.add_documents([document])
    # final split
    split_name = save_path.split('.')[0] + '_' + str(s) +  '.pkl'
    save_knowledge_base(vectorstore, split_name)


def index2vectorstore(index_path, vectorstore_path):
    # deprecated
    '''Reference
    1. https://www.kaggle.com/datasets/jjinho/wikipedia-2023-07-faiss-index/data
    2. https://www.kaggle.com/code/samson8/how-to-create-wikipedia-embeddings
    3. https://www.kaggle.com/datasets/jjinho/wikipedia-20230701?select=x.parquet
    '''
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L12-v2',
        model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    wikipedia_index = faiss.read_index(index_path)
    vectorstore = FAISS(embeddings, wikipedia_index)
    # vectorstore = FAISS(index_path, embeddings)
    with open(vectorstore_path, 'wb') as f:
        pickle.dump(vectorstore, f)


# by default, right answers always come first
def shuffle_data(data_path='data/mc_task.json', new_data_path='data/mc_task_random.json'):
    with open(data_path) as f:
        data = json.load(f)
    new_data = []
    for D in data:
        for key in ['mc1_targets', 'mc2_targets']:
            items = list(D[key].items())
            random.shuffle(items)
            D[key] = dict(items)
        new_data.append(D)

    with open(new_data_path, 'w') as f:
        json.dump(data, f)


def extract_mc1_answer_from_llama_output(text):
    pattern = r'"answer"\s*:\s*(\d+)'
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    else:
        return None
    

def extract_mc2_answer_from_llama_output(text):
    pattern = r'"answer"\s*:\s*\[([^\]]+)\]'
    match = re.search(pattern, text)
    if match:
        numbers_str = match.group(1)
        numbers = [int(num) for num in numbers_str.split(',')]
        return numbers
    else:
        return []
    
def extract_json_from_llama_output(text):
    pattern = r"```json\n([\s\S]+?)\n```"
    matched_json = re.search(pattern, text)
    if matched_json:
        extracted_json = matched_json.group(1)
        return extracted_json
    else:
        # backup plan
        pattern = r"\{.*?\}"
        matched_json = re.search(pattern, text, re.DOTALL)
        if matched_json:
            extracted_json = matched_json.group()
            return extracted_json
        else:
            raise ValueError('No JSON structure found.')
    


if __name__ == '__main__':
    ## create wikipedia knowledge base
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # load_wikipedia_knowledge_base(data_files=None, split=10)
    # merge_knowledge_base('knowledge_base/', '/data/langcao2/knowledge_base/wikipedia.pkl')
    # index2vectorstore('data/wikipedia_202307.index', 'knowledge_base/wikipedia.pkl')


    ## shuffle data
    shuffle_data()


