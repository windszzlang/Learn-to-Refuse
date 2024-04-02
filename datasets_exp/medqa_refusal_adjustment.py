import os
import sys
import json


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from chain import L2R_Chain
from utils import *
from evaluate import *
from proccess_data import *


output_path = 'datasets_exp/output/l2r_m_refusal'

# model = L2R_Chain()
# threshold = L2R_Chain.threshold # default
# threshold = 1e9 # no hard refusal
threshold = 0.6
# threshold = 1 # no hard refusal

L2R_Chain.threshold = threshold
print(L2R_Chain.threshold)



for file in os.listdir(output_path):
    file_path = os.path.join(output_path, file)
    if os.path.isfile(file_path):
        with open(file_path) as f:
            result = json.load(f)
        
        scores, confidences = [], []
        for doc in result['retrieved_documents']:
            scores.append(doc['score'])
            # scores.append(1)
            confidences.append(doc['metadata']['confidence'])
            # confidences.append(1)
            
        new_hard_refusal = not L2R_Chain.judge_relevance(scores, confidences)
        result['hard_refusal'] = new_hard_refusal
        # result['hard_refusal'] = False

        with open(file_path, 'w') as f:
            json.dump(result, f)


data = load_med_qa_data('datasets_exp/data/MedQA.jsonl')
predictions = gather_prediction(output_path, 'json')
pred = [p['answer'] for p in predictions]
gold = [D[2] for D in data]



refusal = [[p['hard_refusal'], p['soft_refusal']] for p in predictions]
# refusal = [[False, False] for p in predictions]
pred, gold = filter_refused(pred, gold, refusal)
res = evaluate_multiple_choice_1(pred, gold)
print(res)
