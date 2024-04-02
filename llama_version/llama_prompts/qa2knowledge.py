QA2KNOWLEDGE_PROMPT_TEMPLATE = '''You are an AI who is responsible for convert a pair of a question and the corresponding answer into a piece of factual knowledge.
I will give you a list of question-answer pairs. in the JSON format. You need to convert all of them them one by one.

You output of a factual knowledge should entirely based on the question-answer pair, which is provided in the "question" and "answer" fields.
Your expression needs to be a declarative sentence and brief to clearly state a fact.

You should retain orginal values in the "q_id" and "confidence" fields.

##### QA Pairs Start #####
{qa_paris}
##### QA Pairs End ######

You must output your response of answered questions in a list in the JSON format of:
```json
[
    {{
        "k_id": 0, (use default value of 0),
        "factual knowledge": "the summarzied factual knowledge based on 'question' and 'answer'",
        "confidence": the degree of confidence in the answer to this question (retain original value)
    }}
        ...
]
```

You should place (```json```) outside of your output of the JSON block.
You must respond me in the JSON format!


Now, you can generate your response:'''