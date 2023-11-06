import os

from utils import *
from chain import L2R_Chain

from env import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


if __name__ == '__main__':
    # data = load_truthfulqa('data/TruthfulQA.csv')

    model = L2R_Chain()

    model.load_knowledge(['knowledge/test_0.txt'])
    model.init_knowledge_base()

    print('Initalization success...')
    while True:
        query = input('User: ')
        answer = model.answer(query)
        print('AI:', answer)