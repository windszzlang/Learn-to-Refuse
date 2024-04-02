MULTIPLE_CHOICE_1_PROMPT_TEMPLATE = '''Question:
{question}

Candidate Answers:
{candidate_answers}

There is only one correct option. Therefore, you must choose the answer that is most likely to be correct from all candidate answers.
You must give an answer of your choice without any explanation.

You cannot give any explanation!
You cannot give any explanation!
You cannot give any explanation!

Your response must follow the json format:
```
{{
    "answer": x
}}
```

Your response '''