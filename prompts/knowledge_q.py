KNOWLEDGE_Q_PROMPT_TEMPLATE = '''You are an AI who is responsible for asking all kinds of questions. These questions must be about a facutal knowledge in the real world.

Here are some examples of generated questions:
{seed_questions}

You should give different questions than the examples above.

You should only output your response of generated questions in a list in the JSON format of:
[
    "question 1",
    "question 2",
    ...
    "question n"
]

Now, you can generate {question_number} questions:'''