KNOWLEDGE_A_PROMPT_TEMPLATE = '''You are an AI who is responsible for answering all kinds of questions. These questions are all about a facutal knowledge in the real world.
I will give you a list of questions in the JSON format. You need to answer these questions one by one.

One important point is that I know you cannot answer every question accurately and even some questions you cannot answer. To deal with this problem, you should give the degree of confidence in your answer to this question at the same time.
The value of confidence should be ranged from 0 to 1.
A confidence value of 1 means you feel your answer is 100 percent correct.
A confidence value of 0.5 means that you think there is a 50 percent chance that your answer is incorrect
A confidence value of 0 indicates that you believe that you cannot give an answer at all, or that the answer you give is totally wrong

You must give me a definite answer and cannot refuse to answer the question. You should use "confidence" to show the confidence of your opinion, not do it in "answer".
You must give me a definite answer and cannot refuse to answer the question. You should use "confidence" to show the confidence of your opinion, not do it in "answer".
You must give me a definite answer and cannot refuse to answer the question. You should use "confidence" to show the confidence of your opinion, not do it in "answer".

##### Questions Start #####
{questions}
##### Questions End ######


You must output your response in a list in the JSON format of:
```json
[
    {{
        "question": "the content of the Question",
        "answer": "one sentence of answer of the Question",
        "confidence": the degree of confidence in the answer to this question (range: 0 to 1)
    }}
    ...
]
```

You should place (```json```) outside of the JSON block.
You must use the JSON format to respond me!

Now, you can generate your response:'''