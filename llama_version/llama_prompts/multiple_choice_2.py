MULTIPLE_CHOICE_2_PROMPT_TEMPLATE = '''Question:
{question}

Candidate Answers:
{candidate_answers}

This is a multiple-answer question, and there can be multiple correct options. Therefore, you need to choose multiple correct answers from all candidate answers.
You must give an answer of your choice without any explanation.

You cannot give any explanation!
You cannot give any explanation!
You cannot give any explanation!

Your response must follow the json format:
```
{{
    "answer": [number_of_correct_option_1, number_of_correct_option_2, ...]
}}
```

Your response '''


# MULTIPLE_CHOICE_2_PROMPT_TEMPLATE = '''Question:
# {question}

# Candidate Answers:
# {candidate_answers}

# This is a multiple-answer question, and there can be multiple correct options. Therefore, you need to choose multiple correct answers from the provided options.
# Your answer should only contain the numbers corresponding to the chosen options. Your answer cannot contain any text.
# Please format your answer as a Python list, like [number_of_correct_option_1, number_of_correct_option_2, ...].
# '''