import os
import json
from chain import L2R_Chain



# output_paths = ['output/l2r_mc1', 'output/l2r_mc2']
# output_paths = ['output/wiki_l2r_mc1', 'output/wiki_l2r_mc2']

output_paths = ['output/gold_l2r_mc1_1.00']

# model = L2R_Chain()
# threshold = L2R_Chain.threshold # default
# threshold = 1e9 # no hard refusal
threshold = 0.75 # no hard refusal
# threshold = 1 # no hard refusal

L2R_Chain.threshold = threshold
print(L2R_Chain.threshold)


for output_path in output_paths:
    for file in os.listdir(output_path):
        file_path = os.path.join(output_path, file)
        if os.path.isfile(file_path):
            with open(file_path) as f:
                result = json.load(f)
            
            scores, confidences = [], []
            for doc in result['retrieved_documents']:
                scores.append(doc['score'])
                confidences.append(doc['metadata']['confidence'])
                
            new_hard_refusal = not L2R_Chain.judge_relevance(scores, confidences)
            result['hard_refusal'] = new_hard_refusal
            # result['hard_refusal'] = False

            with open(file_path, 'w') as f:
                json.dump(result, f)
