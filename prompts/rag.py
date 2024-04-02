RAG_PROMPT_TEMPLATE = '''You can answer this question using the background I have provided you

#### Background Knowledge START ####
{knowledge}
#### Background Knowledge END ####

#### Question Start ####
{question}
#### Question End ####


You must output your response in exactly the following JSON format (which contains four fields: answer):
{{
    "answer": your final answer to this the question (if you cannot give answer, you also need to keep this field with the default value `null`),
}}

Now, you can generate your response:'''

