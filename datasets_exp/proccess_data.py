import json



def letter_to_number(letter):
    return ord(letter) - ord('A') + 1


def load_commonsense_qa_data(data_path='datasets_exp/data/CommonsenseQA.jsonl'):
    data = []
    with open(data_path) as f:
        for line in f.readlines():
            data.append(json.loads(line))
    questions = []
    candidate_answers = []
    answers = []
    for D in data:
        questions.append(D['question']['stem'])

        answer = []
        answer.append(letter_to_number(D['answerKey']))
        answers.append(answer)

        candidate_answer = ''
        answer_list = [choice['text'] for choice in D['question']['choices']]
        for i, ans in enumerate(answer_list):
            candidate_answer += str(i + 1) + ': ' + ans + '\n'
        candidate_answers.append(candidate_answer)

    
    new_data = []
    for q, c, a in zip(questions, candidate_answers, answers):
        new_D = [q, c, a]
        new_data.append(new_D)
        
    # type: list (dict) = [[question, candidate_answer, answers], ...]
    return new_data


def load_med_qa_data(data_path='datasets_exp/data/MedQA.jsonl'):
    data = []
    with open(data_path) as f:
        for line in f.readlines():
            data.append(json.loads(line))
    questions = []
    candidate_answers = []
    answers = []
    for D in data:
        questions.append(D['question'])

        answer = []
        answer.append(letter_to_number(D['answer_idx']))
        answers.append(answer)

        candidate_answer = ''
        answer_list = list(D['options'].values())
        for i, ans in enumerate(answer_list):
            candidate_answer += str(i + 1) + ': ' + ans + '\n'
        candidate_answers.append(candidate_answer)

    
    new_data = []
    for q, c, a in zip(questions, candidate_answers, answers):
        new_D = [q, c, a]
        new_data.append(new_D)
        # print(new_D)
        
    # type: list (dict) = [[question, candidate_answer, answers], ...]
    return new_data



if __name__ == '__main__':
    load_commonsense_qa_data()
    load_med_qa_data()