import random
import pickle
import os
import json
from tqdm import tqdm
from queue import Queue
from typing import List

import torch
# from datasets import load_dataset
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader


from prompts.knowledge_a import KNOWLEDGE_A_PROMPT_TEMPLATE
from prompts.knowledge_q import KNOWLEDGE_Q_PROMPT_TEMPLATE
from prompts.main_qa import MAIN_QA_PROMPT_TEMPLATE
from prompts.qa2knowledge import QA2KNOWLEDGE_PROMPT_TEMPLATE



class L2R_Chain:

    threshold = 0.75

    def __init__(self):
        self.openai_model = 'gpt-3.5-turbo-0613' # 'gpt-4'
        self.temperature = 0

        self.main_qa_chain = self._init_main_qa_chain()
        self.knowledge_question_chain = self._init_knowledge_question_chain()
        self.knowledge_answer_chain = self._init_knowledge_answer_chain()
        self.qa2knowledge_chain = self._init_qa2knowledge_chain()
        
        self.knowledge_list = []
        self.knowledge_base = None # retriever
        self.retrieval_top_k = 4
        self.local_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _init_main_qa_chain(self) -> LLMChain:
        llm = ChatOpenAI(model_name=self.openai_model, temperature=self.temperature)
        prompt = PromptTemplate(input_variables=['knowledge', 'question'], template=MAIN_QA_PROMPT_TEMPLATE)
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain

    def _init_knowledge_question_chain(self) -> LLMChain:
        llm = ChatOpenAI(model_name=self.openai_model, temperature=self.temperature)
        prompt = PromptTemplate(input_variables=['seed_questions', 'question_number'], template=KNOWLEDGE_Q_PROMPT_TEMPLATE)
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain
    
    def _init_knowledge_answer_chain(self) -> LLMChain:
        llm = ChatOpenAI(model_name=self.openai_model, temperature=self.temperature)
        prompt = PromptTemplate(input_variables=['questions'], template=KNOWLEDGE_A_PROMPT_TEMPLATE)
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain

    def _init_qa2knowledge_chain(self) -> LLMChain:
        llm = ChatOpenAI(model_name=self.openai_model, temperature=self.temperature)
        prompt = PromptTemplate(input_variables=['qa_paris'], template=QA2KNOWLEDGE_PROMPT_TEMPLATE)
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain
    
    '''
    knowledge_base: a binary file that represent knowledge and its retrieval engine.
    knowledge: a txt file that consists of a list of knowledge with confidence
    '''
    def load_knowledge(self, knowledge_file_list: List, include_confidence=True, load_part=None):
        for file_path in knowledge_file_list:
            i = 0
            with open(file_path) as f:
                for line in f.readlines():
                    if line == '\n':
                        continue
                    if include_confidence:
                        confidence, knowledge = line.strip().split('\t')
                    else:
                        confidence, knowledge = 1, line.strip()
                    self.knowledge_list.append({
                        'k_id': len(self.knowledge_list),
                        'knowledge': knowledge,
                        'confidence': float(confidence)
                    })
                    i += 1
                    if load_part and i >= load_part:
                        break

    def save_knowledge(self, knowledge_file_path, include_confidence=True, separator='\n\n'):
        with open(knowledge_file_path, 'w') as f:
            for i, k in enumerate(self.knowledge_list):
                if include_confidence:
                    f.write(str(k['confidence']) + '\t' + k['knowledge'])
                else:
                    f.write(k['knowledge'])
                if i != len(self.knowledge_list) - 1:
                    f.write(separator)

    def init_knowledge_base(self):
        separator = '\nTMP-SEPARATOR\n'
        self.save_knowledge('knowledge/tmp.txt', include_confidence=False, separator=separator)
        loader = UnstructuredFileLoader('knowledge/tmp.txt')
        raw_documents = loader.load()
        # print(raw_documents)
        # return
        # CharacterTextSplitter will only split on separator
        text_splitter = CharacterTextSplitter(
            separator=separator, 
            chunk_size=0, # if splitting is possible (has one separator) and current chunk is greater than chunk_size, it will be split .
            chunk_overlap=0,
            length_function=len
        )
        documents = text_splitter.split_documents(raw_documents)

        # add confidence to metadata
        for K, D in zip(self.knowledge_list, documents):
            D.metadata['confidence'] = K['confidence']

        # print(raw_documents)
        # print(documents)

        # embeddings = OpenAIEmbeddings()
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-mpnet-base-v2',
            model_kwargs = {'device': self.local_device}
        )
        vectorstore = FAISS.from_documents(documents, embeddings)
        self.knowledge_base = vectorstore

    def load_knowledge_base(self, knowledge_base_path):
        with open(knowledge_base_path, 'rb') as f:
            self.knowledge_base = pickle.load(f)
        # retriever = VectorStoreRetriever(vectorstore=vectorstore)

    def save_knowledge_base(self, knowledge_base_path):
        with open(knowledge_base_path, 'wb') as f:
            pickle.dump(self.knowledge_base, f)

    def automatic_knowledge_enrich(self, number=20, batch_size=10, cache_dir='cache'):
        # step 1: get question
        question_list = []
        for i in range(int(number / batch_size) + 1):
            if question_list == []:
                seed_questions = []
            else:
                seed_questions = random.sample(question_list, batch_size)
            questions = self.knowledge_question_chain.predict(seed_questions=json.dumps(seed_questions), question_number=batch_size)
            question_list.extend(json.loads(questions))
        # print(question_list)
        # get answer and knowledge
        input('Confirm to go to the next step...')
        self.enrich_knowledge_from_truthfulqa(question_list, batch_size=batch_size, cache_dir=cache_dir)
        return

    def enrich_knowledge_from_truthfulqa(self, question_list, batch_size=10, cache_dir='cache'):
        # step 1: get answer
        '''
        "q_id": the id of the question,
        "question": the content of the question,
        "answer": the answer of the question,
        "confidence": the degree of confidence in the answer to this question (range: 0 to 1)
        '''
        print('Generate answers...')
        new_question_list = []
        for i, q in enumerate(question_list):
            new_question_list.append({'question': q})
        
        qa_pairs = []
        questions = []
        j = 0
        for i in tqdm(range(len(new_question_list))):
            questions.append(new_question_list[i])
            j += 1
            if j >= batch_size or i == len(new_question_list) - 1:
                cache_path = cache_dir+ '/answer/' + str(i) +  '.json'
                if not os.path.exists(cache_path):
                    try:
                        model_output = self.knowledge_answer_chain.predict(questions=json.dumps(questions))
                        result = json.loads(model_output)
                        with open(cache_path , 'w') as f:
                            json.dump(result, f)
                    except Exception as e:
                        print(e)
                        print(model_output)
                questions = []
                j = 0
        # read cache
        for file in os.listdir(cache_dir + '/answer/'):
            file_path = os.path.join(cache_dir + '/answer/', file)
            if os.path.isfile(file_path):
                result = json.load(open(file_path))
                qa_pairs.extend(result)

        # step 2: get knowledge
        '''
        "q_id": the id of the question (retain original value),
        "factual knowledge": the summarzied factual knowledge based on 'question' and 'answer',
        "confidence": the degree of confidence in the answer to this question (retain original value)
        '''
        input('Confirm to go to the next step...')
        print('Generate knowledge...')
        new_knowledge = []
        tmp_qa_pairs = []
        j = 0
        for i in tqdm(range(len(qa_pairs))):
            tmp_qa_pairs.append(qa_pairs[i])
            j += 1
            if j >= batch_size or i == len(qa_pairs) - 1:
                cache_path = cache_dir + '/knowledge/' + str(i) +  '.json'
                if not os.path.exists(cache_path):
                    try:
                        model_output = self.qa2knowledge_chain.predict(qa_paris=json.dumps(tmp_qa_pairs))
                        result = json.loads(model_output)
                        with open(cache_path , 'w') as f:
                            json.dump(result, f)
                    except Exception as e:
                        print(e)
                        print(model_output)
                tmp_qa_pairs = []
                j = 0
        # read cache
        for file in os.listdir(cache_dir + '/knowledge/'):
            file_path = os.path.join(cache_dir + '/knowledge/', file)
            if os.path.isfile(file_path):
                with open(file_path) as f:
                    new_knowledge = json.load(f)
                for k in new_knowledge:
                    self.knowledge_list.append({
                        'k_id': len(self.knowledge_list),
                        'knowledge': k['factual knowledge'],
                        'confidence': k['confidence'],
                    })
    
    def enrich_gold_knowledge_from_truthfulqa(self, question_list, answer_list, batch_size=10):
        # step 1: get answer
        qa_pairs = []
        for question, answer in zip(question_list, answer_list):
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "confidence": 1.0
            })
        # step 2: get knowledge
        '''
        "q_id": the id of the question (retain original value),
        "factual knowledge": the summarzied factual knowledge based on 'question' and 'answer',
        "confidence": the degree of confidence in the answer to this question (retain original value)
        '''
        print('Generate knowledge...')
        new_knowledge = []
        tmp_qa_pairs = []
        j = 0
        for i in tqdm(range(len(qa_pairs))):
            tmp_qa_pairs.append(qa_pairs[i])
            j += 1
            if j >= batch_size or i == len(qa_pairs) - 1:
                cache_path = 'cache/gold_knowledge/' + str(i) +  '.json'
                if not os.path.exists(cache_path):
                    try:
                        model_output = self.qa2knowledge_chain.predict(qa_paris=json.dumps(tmp_qa_pairs))
                        result = json.loads(model_output)
                        with open(cache_path , 'w') as f:
                            json.dump(result, f)
                    except Exception as e:
                        print(e)
                        print(model_output)
                tmp_qa_pairs = []
                j = 0
        # read cache
        for file in os.listdir('cache/gold_knowledge/'):
            file_path = os.path.join('cache/gold_knowledge/', file)
            if os.path.isfile(file_path):
                with open(file_path) as f:
                    new_knowledge = json.load(f)
                for k in new_knowledge:
                    self.knowledge_list.append({
                        'k_id': len(self.knowledge_list),
                        'knowledge': k['factual knowledge'],
                        'confidence': k['confidence'],
                    })

    @classmethod
    def judge_relevance(self, scores: List, confidences: List):
        '''Hard refusal function design
        '''
        # overall_score = sum([(1-a) * b for a, b in zip(scores, confidences)]) / len(scores)
        # overall_score = max([(1-a) * b for a, b in zip(scores, confidences)])
        overall_score = min([a / (b + 1e-9) for a, b in zip(scores, confidences)])
        results = overall_score < self.threshold

        return results # answerablity: true: can answer, false: cannot answer

    def answer(self, query: str, question=None, full_output=False, wikipedia=False, original_output=False) -> str:
        '''
        Sometimes question is not the same as query, question may be wrapped in mutlple choice framework. It is used to query knowledge base.
        '''
        tmp_knowledge = []
        scores = []
        confidences = []
        retrieved_documents = []
        # l2 euclidean distance score [0,+), 0 means the same, the smaller is the better
        if question: # original question
            results_with_scores = self.knowledge_base.similarity_search_with_score(question, k=self.retrieval_top_k)
        else: # wrapped query for chain input
            results_with_scores = self.knowledge_base.similarity_search_with_score(query, k=self.retrieval_top_k)
        # print(question)
        for doc, score in results_with_scores:
            # print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
            tmp_knowledge.append(doc.page_content)
            scores.append(score)
            confidences.append(doc.metadata['confidence'])
            retrieved_documents.append({
                'content': doc.page_content, 
                'metadata': doc.metadata,
                'score': float(score)
            })

        tmp_knowledge = json.dumps(tmp_knowledge)
        # print(tmp_knowledge)
        # print(scores)

        if self.judge_relevance(scores, confidences):
            hard_refusal = False
        else:
            hard_refusal = True

        result = self.main_qa_chain.predict(knowledge=tmp_knowledge, question=query)
        result = json.loads(result)
        result['retrieved_documents'] = retrieved_documents
        result['question'] = query

        if original_output:
            return result

        result['soft_refusal'] = not result['CAN_ANSWER']
        if wikipedia:
            result['hard_refusal'] = False
        else:
            result['hard_refusal'] = hard_refusal
        # print(result)
        result.pop('CAN_ANSWER')
        if 'answer' not in result.keys():
            result['answer'] = ''
        if full_output:
            return result
        else:
            return result['answer']



if __name__ == '__main__':
    # for qualitative experiment
    from env import OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    model = L2R_Chain()
    model.load_knowledge(['knowledge/test_0.txt'])
    model.init_knowledge_base()

    print('Initalization success...')
    while True:
        query = input('User: ')
        answer = model.answer(query)
        print('AI:', answer)