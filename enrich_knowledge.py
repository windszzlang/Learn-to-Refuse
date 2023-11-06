import os
from tqdm import tqdm

from utils import *
from chain import L2R_Chain

from env import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

model = L2R_Chain()

model.automatic_knowledge_enrich()