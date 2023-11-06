MULTIPLE_CHOICE_2_PROMPT_TEMPLATE = '''Question:
{question}

Candidate Answers:
{candidate_answers}

This is a multiple-answer question, and there can be multiple correct options. Therefore, you need to choose multiple correct answers from all candidate answers.
Your answer should only contain numbers of the chosen options. Your answer cannot contain any textual content.
The format of your answer must follow a list in Python like [number_of_correct_option_1, number_of_correct_option_2, ...].
'''

# MULTIPLE_CHOICE_2_PROMPT_TEMPLATE = '''Question:
# {question}

# Candidate Answers:
# {candidate_answers}

# This is a multiple-answer question, and there can be multiple correct options. Therefore, you need to choose multiple correct answers from the provided options.
# Your answer should only contain the numbers corresponding to the chosen options. Your answer cannot contain any text.
# Please format your answer as a Python list, like [number_of_correct_option_1, number_of_correct_option_2, ...].
# '''